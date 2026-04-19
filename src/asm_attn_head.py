#!/usr/bin/env python3
"""
6502 attn_head + parity test against op_attn_head.

computes one attention head for one query position:
    scores[t] = Σ_j q[j] * k[t, j] (int32, t=0..NKEYS-1)
    sf[t]     = scores[t] >> 14
    m         = max(sf)
    w[t]      = EXP_LUT[clamp(m - sf[t], 0, 127)]  (u8)
    w_sum     = Σ w[t]
    out[j]    = sat16((Σ_t w[t] * v[t, j]) / w_sum)

calling convention (zero page):
  $F0/$F1  QP      query pointer
  $F2/$F3  KB      K base
  $F4/$F5  VB      V base
  $F6/$F7  OP      output pointer
  $F8      NKEYS   number of keys (1..SL=20)
  $F9      HEAD    head index (0..NH-1=3)

internal (reuses smul16 slots TMP/SRC16/SIGN/PROD from asm_rms_norm layout):
  $10..$13  ACC32  (reuses matvec's ACC32 slot for final division acc)
  $14       TIDX   key iteration index
  $15       J      output dim index
  $16/$17   KROW   current K row pointer
  $18/$19   TMP    (shared with smul16)
  $1A/$1B   SRC16  (shared with smul16)
  $1C       SIGN   (shared)
  $20..$23  PROD   (shared)
  $30/$31   MAXSF  max sf  (int16)
  $32/$33   WSUM   weights sum (u16)
  $34..$37  T32    scratch (reused as divide remainder)
  $38..$3B  SCR_A  scratch (reused by divide)
  $3C..$3F  SCR_B  scratch

external buffers (pointers in zp):
  $FA/$FB  SCORES_P  base of SCORES buffer (int32 × NKEYS = 80 bytes max)
  $FC/$FD  WTS_P     base of WEIGHTS buffer (u8 × NKEYS = 20 bytes max)
  $FE/$FF  ELUT_P    pointer to EXP_LUT (128 bytes)
"""
import numpy as np
from assembler import (
    CodeBuilder, run_subroutine,
    LDA_imm, LDX_imm, LDY_imm, LDA_zp, LDX_zp, LDY_zp,
    STA_zp, LDA_indY, STA_indY, LDA_absX, LDA_absY,
    ADC_imm, ADC_zp, SBC_imm, SBC_zp, EOR_imm, CMP_imm,
    INC_zp, DEC_zp, ASL_zp, ROL_zp, ROR_zp, LSR_zp, ASL_A, LSR_A,
    CLC, SEC, RTS, BRK, PHA, PLA, TAY, TYA, TXA, TAX, DEX, INX,
)
from asm_rms_norm import build_smul16, TMP, SRC16, SIGN, PROD

# Zero page
QP, KB, VB, OP = 0x40, 0x42, 0x44, 0x46
NKEYS, HEAD = 0x48, 0x49
SCORES_P, WTS_P, ELUT_P = 0x4A, 0x4C, 0x4E

ACC32 = 0x10
TIDX  = 0x14
J     = 0x15
KROW  = 0x16
MAXSF = 0x30
WSUM  = 0x32
T32   = 0x34
SCR_A = 0x38
SCR_B = 0x3C

STRIDE = 64     # bytes per position in K/V buffers (ED*2)
HD     = 8      # per-head dim


def INY_op(cb): cb.emit(0xC8)
def DEY_op(cb): cb.emit(0x88)


def build_sdiv_i32_u16(cb):
    """Signed int32 / unsigned u16 → signed int16 (saturating).

    Inputs:
      T32[0..3]   dividend (signed int32)
      WSUM[0..1]  divisor  (unsigned u16, nonzero)
    Output:
      SCR_B[0..1] quotient (signed int16, saturated)

    Strategy: track sign, absolute the dividend, do unsigned 32÷16 → 32 via
    shift-and-subtract, negate result if sign was set, saturate to int16.
    """
    cb.label('sdiv')
    # Save sign
    LDA_zp(cb, T32 + 3)
    cb.emit_branch(0x10, '_div_pos')        # BPL
    # Negate T32 (two's complement)
    LDA_zp(cb, T32);     EOR_imm(cb, 0xFF); STA_zp(cb, T32)
    LDA_zp(cb, T32 + 1); EOR_imm(cb, 0xFF); STA_zp(cb, T32 + 1)
    LDA_zp(cb, T32 + 2); EOR_imm(cb, 0xFF); STA_zp(cb, T32 + 2)
    LDA_zp(cb, T32 + 3); EOR_imm(cb, 0xFF); STA_zp(cb, T32 + 3)
    CLC(cb)
    LDA_zp(cb, T32);     ADC_imm(cb, 1); STA_zp(cb, T32)
    LDA_zp(cb, T32 + 1); ADC_imm(cb, 0); STA_zp(cb, T32 + 1)
    LDA_zp(cb, T32 + 2); ADC_imm(cb, 0); STA_zp(cb, T32 + 2)
    LDA_zp(cb, T32 + 3); ADC_imm(cb, 0); STA_zp(cb, T32 + 3)
    LDA_imm(cb, 1); STA_zp(cb, SIGN)
    cb.emit_jmp('_div_start')
    cb.label('_div_pos')
    LDA_imm(cb, 0); STA_zp(cb, SIGN)

    cb.label('_div_start')
    # Quotient in SCR_A[0..3], remainder initially 0 in SCR_B[0..3]
    LDA_imm(cb, 0)
    STA_zp(cb, SCR_A);     STA_zp(cb, SCR_A + 1)
    STA_zp(cb, SCR_A + 2); STA_zp(cb, SCR_A + 3)
    STA_zp(cb, SCR_B);     STA_zp(cb, SCR_B + 1)
    STA_zp(cb, SCR_B + 2); STA_zp(cb, SCR_B + 3)

    # 32-iteration restoring division: shift dividend left into remainder,
    # trial-subtract divisor (16-bit, zero-extended) from remainder, if no
    # borrow commit and set quotient bit.
    LDX_imm(cb, 32)
    cb.label('_div_lp')
    # Shift T32 left 1, with bit falling off the top into SCR_B[0]
    ASL_zp(cb, T32); ROL_zp(cb, T32 + 1); ROL_zp(cb, T32 + 2); ROL_zp(cb, T32 + 3)
    ROL_zp(cb, SCR_B); ROL_zp(cb, SCR_B + 1); ROL_zp(cb, SCR_B + 2); ROL_zp(cb, SCR_B + 3)
    # Shift quotient left 1
    ASL_zp(cb, SCR_A); ROL_zp(cb, SCR_A + 1); ROL_zp(cb, SCR_A + 2); ROL_zp(cb, SCR_A + 3)
    # Trial subtract WSUM (16-bit) from SCR_B (32-bit)
    SEC(cb)
    LDA_zp(cb, SCR_B);     SBC_zp(cb, WSUM);     STA_zp(cb, PROD)
    LDA_zp(cb, SCR_B + 1); SBC_zp(cb, WSUM + 1); STA_zp(cb, PROD + 1)
    LDA_zp(cb, SCR_B + 2); SBC_imm(cb, 0);       STA_zp(cb, PROD + 2)
    LDA_zp(cb, SCR_B + 3); SBC_imm(cb, 0);       STA_zp(cb, PROD + 3)
    cb.emit_branch(0x90, '_div_nogo')       # BCC: borrow → don't commit
    # Commit subtraction and set quotient bit 0
    LDA_zp(cb, PROD);     STA_zp(cb, SCR_B)
    LDA_zp(cb, PROD + 1); STA_zp(cb, SCR_B + 1)
    LDA_zp(cb, PROD + 2); STA_zp(cb, SCR_B + 2)
    LDA_zp(cb, PROD + 3); STA_zp(cb, SCR_B + 3)
    LDA_zp(cb, SCR_A); cb.emit(0x09, 0x01); STA_zp(cb, SCR_A)    # ORA #1
    cb.label('_div_nogo')
    DEX(cb)
    cb.emit_branch_far(0xD0, '_div_lp')

    # Negate quotient if sign was set. But first: for floor-division semantics
    # (matching Python's //), if the result is negative AND the remainder
    # (SCR_B) is nonzero, we need to add 1 to the unsigned quotient before
    # negating. This makes the result round toward -∞ instead of toward 0.
    LDA_zp(cb, SIGN)
    cb.emit_branch(0xF0, '_div_sat')       # BEQ: positive → skip
    # Check remainder != 0
    LDA_zp(cb, SCR_B)
    cb.emit(0x05, SCR_B + 1)     # ORA SCR_B+1
    cb.emit(0x05, SCR_B + 2)
    cb.emit(0x05, SCR_B + 3)
    cb.emit_branch(0xF0, '_div_no_adj')    # BEQ: remainder is 0 → no adjustment
    # Increment quotient by 1 (unsigned)
    INC_zp(cb, SCR_A)
    cb.emit_branch(0xD0, '_div_no_adj')
    INC_zp(cb, SCR_A + 1)
    cb.emit_branch(0xD0, '_div_no_adj')
    INC_zp(cb, SCR_A + 2)
    cb.emit_branch(0xD0, '_div_no_adj')
    INC_zp(cb, SCR_A + 3)
    cb.label('_div_no_adj')
    # Negate quotient
    LDA_zp(cb, SCR_A);     EOR_imm(cb, 0xFF); STA_zp(cb, SCR_A)
    LDA_zp(cb, SCR_A + 1); EOR_imm(cb, 0xFF); STA_zp(cb, SCR_A + 1)
    LDA_zp(cb, SCR_A + 2); EOR_imm(cb, 0xFF); STA_zp(cb, SCR_A + 2)
    LDA_zp(cb, SCR_A + 3); EOR_imm(cb, 0xFF); STA_zp(cb, SCR_A + 3)
    CLC(cb)
    LDA_zp(cb, SCR_A);     ADC_imm(cb, 1); STA_zp(cb, SCR_A)
    LDA_zp(cb, SCR_A + 1); ADC_imm(cb, 0); STA_zp(cb, SCR_A + 1)
    LDA_zp(cb, SCR_A + 2); ADC_imm(cb, 0); STA_zp(cb, SCR_A + 2)
    LDA_zp(cb, SCR_A + 3); ADC_imm(cb, 0); STA_zp(cb, SCR_A + 3)

    cb.label('_div_sat')
    # Saturate to int16. Dispatch on SCR_A[3] sign bit.
    LDA_zp(cb, SCR_A + 3)
    cb.emit_branch(0x30, '_div_neg_chk')   # BMI
    # Positive: SCR_A[2] and [3] must both be 0, and SCR_A[1] bit 7 must be 0
    LDA_zp(cb, SCR_A + 2)
    cb.emit_branch_far(0xD0, '_div_sat_pos')
    LDA_zp(cb, SCR_A + 3)
    cb.emit_branch_far(0xD0, '_div_sat_pos')
    LDA_zp(cb, SCR_A + 1)
    cb.emit_branch_far(0x30, '_div_sat_pos')
    cb.emit_jmp('_div_done')
    cb.label('_div_neg_chk')
    LDA_zp(cb, SCR_A + 2); CMP_imm(cb, 0xFF)
    cb.emit_branch_far(0xD0, '_div_sat_neg')
    LDA_zp(cb, SCR_A + 3); CMP_imm(cb, 0xFF)
    cb.emit_branch_far(0xD0, '_div_sat_neg')
    LDA_zp(cb, SCR_A + 1)
    cb.emit_branch_far(0x10, '_div_sat_neg')
    cb.emit_jmp('_div_done')
    cb.label('_div_sat_pos')
    LDA_imm(cb, 0xFF); STA_zp(cb, SCR_A)
    LDA_imm(cb, 0x7F); STA_zp(cb, SCR_A + 1)
    cb.emit_jmp('_div_done')
    cb.label('_div_sat_neg')
    LDA_imm(cb, 0x00); STA_zp(cb, SCR_A)
    LDA_imm(cb, 0x80); STA_zp(cb, SCR_A + 1)
    cb.label('_div_done')
    RTS(cb)


def build_attn_head(cb: CodeBuilder):
    """Assemble the attn_head routine. Requires smul16 and sdiv to also be
    in the builder."""

    cb.label('attn_head')

    # ─── Step 1: compute scores[t] = Σ q[j] * k[t, j] for t in 0..NKEYS-1 ──
    # KROW = KB + HEAD*16 (skip to this head's slot inside position 0)
    LDA_zp(cb, HEAD)
    ASL_A(cb); ASL_A(cb); ASL_A(cb); ASL_A(cb)  # *16
    CLC(cb)
    ADC_zp(cb, KB); STA_zp(cb, KROW)
    LDA_zp(cb, KB + 1); ADC_imm(cb, 0); STA_zp(cb, KROW + 1)
    # TIDX = 0
    LDA_imm(cb, 0); STA_zp(cb, TIDX)

    cb.label('_ah_score_lp')
    # Clear 32-bit dot accumulator (in T32)
    LDA_imm(cb, 0)
    STA_zp(cb, T32); STA_zp(cb, T32 + 1)
    STA_zp(cb, T32 + 2); STA_zp(cb, T32 + 3)
    # J = 0 (loop over HD)
    STA_zp(cb, J)

    cb.label('_ah_dot_lp')
    # Load q[J] into TMP
    LDA_zp(cb, J); ASL_A(cb); TAY(cb)
    LDA_indY(cb, QP); STA_zp(cb, TMP)
    INY_op(cb)
    LDA_indY(cb, QP); STA_zp(cb, TMP + 1)
    # Load k[TIDX][J] via KROW,Y (Y = J*2)
    LDA_zp(cb, J); ASL_A(cb); TAY(cb)
    LDA_indY(cb, KROW); STA_zp(cb, SRC16)
    INY_op(cb)
    LDA_indY(cb, KROW); STA_zp(cb, SRC16 + 1)
    # Multiply TMP × SRC16 → PROD (32-bit)
    cb.emit_jsr('smul16')
    # T32 += PROD (signed 32-bit add)
    CLC(cb)
    LDA_zp(cb, T32);     ADC_zp(cb, PROD);     STA_zp(cb, T32)
    LDA_zp(cb, T32 + 1); ADC_zp(cb, PROD + 1); STA_zp(cb, T32 + 1)
    LDA_zp(cb, T32 + 2); ADC_zp(cb, PROD + 2); STA_zp(cb, T32 + 2)
    LDA_zp(cb, T32 + 3); ADC_zp(cb, PROD + 3); STA_zp(cb, T32 + 3)
    INC_zp(cb, J)
    LDA_zp(cb, J); CMP_imm(cb, HD)
    cb.emit_branch_far(0x90, '_ah_dot_lp')

    # done with dot product for TIDX. Shift T32 right by 17 to get sf (int16)
    # using 16 arithmetic shifts + 1 more = 17 total
    LDX_imm(cb, 14)
    cb.label('_ah_sf_shift')
    LDA_zp(cb, T32 + 3); CMP_imm(cb, 0x80)   # sign bit -> C
    ROR_zp(cb, T32 + 3); ROR_zp(cb, T32 + 2); ROR_zp(cb, T32 + 1); ROR_zp(cb, T32)
    DEX(cb)
    cb.emit_branch_far(0xD0, '_ah_sf_shift')
    # store sf (as int16) in SCORES buffer at TIDX*2
    LDA_zp(cb, TIDX); ASL_A(cb); TAY(cb)
    LDA_zp(cb, T32);     STA_indY(cb, SCORES_P)
    INY_op(cb)
    LDA_zp(cb, T32 + 1); STA_indY(cb, SCORES_P)

    # advance KROW by STRIDE (64 bytes) for next pos
    CLC(cb)
    LDA_zp(cb, KROW);     ADC_imm(cb, STRIDE); STA_zp(cb, KROW)
    LDA_zp(cb, KROW + 1); ADC_imm(cb, 0);      STA_zp(cb, KROW + 1)
    # ++TIDX
    INC_zp(cb, TIDX)
    LDA_zp(cb, TIDX)
    cb.emit(0xC5, NKEYS)         # CMP zp NKEYS
    cb.emit_branch_far(0x90, '_ah_score_lp')

    # find max of sf[0..NKEYS-1]
    # init MAXSF = sf[0]
    LDY_imm(cb, 0)
    LDA_indY(cb, SCORES_P); STA_zp(cb, MAXSF)
    INY_op(cb)
    LDA_indY(cb, SCORES_P); STA_zp(cb, MAXSF + 1)
    # TIDX = 1
    LDA_imm(cb, 1); STA_zp(cb, TIDX)

    cb.label('_ah_max_lp')
    LDA_zp(cb, TIDX)
    cb.emit(0xC5, NKEYS)         # CMP NKEYS
    cb.emit_branch_far(0xB0, '_ah_max_done')   # BCS: TIDX >= NKEYS -> done
    # load sf[TIDX] into PROD[0..1] (scratch)
    LDA_zp(cb, TIDX); ASL_A(cb); TAY(cb)
    LDA_indY(cb, SCORES_P); STA_zp(cb, PROD)
    INY_op(cb)
    LDA_indY(cb, SCORES_P); STA_zp(cb, PROD + 1)
    # signed compare: is (PROD - MAXSF) > 0?
    # use SBC then check sign of result for overflow-free signed compare:
    # a > b (signed) iff (a-b) positive and no signed overflow.
    # simpler: if sign bits differ, the positive one is bigger. if equal->low bits decide
    # Signed 16-bit compare: PROD vs MAXSF. If PROD > MAXSF, update MAXSF.
    # compare high bytes as signed (XOR trick), low bytes as unsigned.
    LDA_zp(cb, PROD + 1)
    cb.emit(0xC5, MAXSF + 1)     # CMP MAXSF+1 (unsigned compare of high bytes)
    cb.emit_branch(0xD0, '_ah_hi_ne')  # BNE: high bytes differ
    # high bytes equal: compare lows unsigned
    LDA_zp(cb, PROD); cb.emit(0xC5, MAXSF)
    cb.emit_branch_far(0x90, '_ah_max_next')   # BCC: PROD < MAXSF
    cb.emit_branch_far(0xF0, '_ah_max_next')   # BEQ: equal, no update
    cb.emit_jmp('_ah_max_update')
    cb.label('_ah_hi_ne')
    # high bytes differ... we want signed comparison. Do it via:
    # signed(PROD+1) > signed(MAXSF+1) iff
    #   (PROD+1 ^ 0x80) > (MAXSF+1 ^ 0x80) unsigned
    LDA_zp(cb, PROD + 1); EOR_imm(cb, 0x80); STA_zp(cb, PROD + 2)  # scratch
    LDA_zp(cb, MAXSF + 1); EOR_imm(cb, 0x80); STA_zp(cb, PROD + 3) # scratch
    LDA_zp(cb, PROD + 2); cb.emit(0xC5, PROD + 3)
    cb.emit_branch_far(0x90, '_ah_max_next')   # BCC: PROD < MAXSF signed
    # fall through: PROD > MAXSF (or equal, but we already know != here :) )
    cb.label('_ah_max_update')
    LDA_zp(cb, PROD);     STA_zp(cb, MAXSF)
    LDA_zp(cb, PROD + 1); STA_zp(cb, MAXSF + 1)

    cb.label('_ah_max_next')
    INC_zp(cb, TIDX)
    cb.emit_jmp('_ah_max_lp')
    cb.label('_ah_max_done')

    # weights[t] = EXP_LUT[clamp(MAXSF - sf[t], 0, 127)]
    # also accumulate WSUM.
    LDA_imm(cb, 0); STA_zp(cb, WSUM); STA_zp(cb, WSUM + 1)
    STA_zp(cb, TIDX)

    cb.label('_ah_wt_lp')
    # load sf[TIDX] into PROD[0..1]
    LDA_zp(cb, TIDX); ASL_A(cb); TAY(cb)
    LDA_indY(cb, SCORES_P); STA_zp(cb, PROD)
    INY_op(cb)
    LDA_indY(cb, SCORES_P); STA_zp(cb, PROD + 1)
    # delta = MAXSF - sf[TIDX], 16-bit signed subtract
    SEC(cb)
    LDA_zp(cb, MAXSF);     SBC_zp(cb, PROD);     STA_zp(cb, PROD + 2)
    LDA_zp(cb, MAXSF + 1); SBC_zp(cb, PROD + 1); STA_zp(cb, PROD + 3)
    # Iif delta < 0 (PROD+3 bit 7 set) or >= 128 (PROD+3 != 0 or PROD+2 bit 7 set),
    # clamp to 127. We clamp aggressively: if PROD+3 != 0, delta is >=256 or <0
    LDA_zp(cb, PROD + 3)
    cb.emit_branch(0xF0, '_ah_delta_lo_ok')    # BEQ: high byte zero, normal
    # high byte nonzero — if it's 0xFF and PROD+2 bit7 set, delta is negative (shouldn't
    # happen but clamp to 0); otherwise clamp to 127
    CMP_imm(cb, 0xFF)
    cb.emit_branch(0xD0, '_ah_clamp_hi')       # BNE: high != FF -> positive overflow
    # high byte is FF, treat as negative -> delta 0
    LDA_imm(cb, 0); STA_zp(cb, PROD + 2)
    cb.emit_jmp('_ah_delta_ready')
    cb.label('_ah_clamp_hi')
    LDA_imm(cb, 127); STA_zp(cb, PROD + 2)
    cb.emit_jmp('_ah_delta_ready')
    cb.label('_ah_delta_lo_ok')
    # PROD+3 == 0. If PROD+2 >= 128, clamp to 127.
    LDA_zp(cb, PROD + 2)
    cb.emit_branch(0x10, '_ah_delta_ready')   # BPL: already in [0,127]
    LDA_imm(cb, 127); STA_zp(cb, PROD + 2)
    cb.label('_ah_delta_ready')
    # look up EXP_LUT[delta]
    LDA_zp(cb, PROD + 2); TAY(cb)
    LDA_indY(cb, ELUT_P)
    # store to weights buffer
    LDY_zp(cb, TIDX)
    STA_indY(cb, WTS_P)
    # add to WSUM (A still holds the weight)
    CLC(cb)
    ADC_zp(cb, WSUM); STA_zp(cb, WSUM)
    LDA_zp(cb, WSUM + 1); ADC_imm(cb, 0); STA_zp(cb, WSUM + 1)
    INC_zp(cb, TIDX)
    LDA_zp(cb, TIDX)
    cb.emit(0xC5, NKEYS)
    cb.emit_branch_far(0x90, '_ah_wt_lp')

    # if WSUM == 0 set to 1
    LDA_zp(cb, WSUM); cb.emit(0x05, WSUM + 1)  # ORA
    cb.emit_branch(0xD0, '_ah_wsum_ok')
    LDA_imm(cb, 1); STA_zp(cb, WSUM)
    cb.label('_ah_wsum_ok')

    # step 5+6.. for each output dim j, out[j] = sat16(Σ w[t]*v[t,j] / WSUM)
    LDA_imm(cb, 0); STA_zp(cb, J)

    cb.label('_ah_out_lp')
    # clear T32 (acc)
    LDA_imm(cb, 0)
    STA_zp(cb, T32); STA_zp(cb, T32 + 1)
    STA_zp(cb, T32 + 2); STA_zp(cb, T32 + 3)

    # KROW := VB + HEAD*16 (VROW actually; reusing KROW slot)
    LDA_zp(cb, HEAD)
    ASL_A(cb); ASL_A(cb); ASL_A(cb); ASL_A(cb)
    CLC(cb)
    ADC_zp(cb, VB); STA_zp(cb, KROW)
    LDA_zp(cb, VB + 1); ADC_imm(cb, 0); STA_zp(cb, KROW + 1)

    LDA_imm(cb, 0); STA_zp(cb, TIDX)

    cb.label('_ah_vt_lp')
    # load w[TIDX] as u8
    LDY_zp(cb, TIDX)
    LDA_indY(cb, WTS_P)
    STA_zp(cb, TMP)
    LDA_imm(cb, 0); STA_zp(cb, TMP + 1)    # zero-extend weight
    # load v[TIDX][J] via KROW,Y (Y = J*2)
    LDA_zp(cb, J); ASL_A(cb); TAY(cb)
    LDA_indY(cb, KROW); STA_zp(cb, SRC16)
    INY_op(cb)
    LDA_indY(cb, KROW); STA_zp(cb, SRC16 + 1)
    cb.emit_jsr('smul16')
    # T32 += PROD (signed 32-bit)
    CLC(cb)
    LDA_zp(cb, T32);     ADC_zp(cb, PROD);     STA_zp(cb, T32)
    LDA_zp(cb, T32 + 1); ADC_zp(cb, PROD + 1); STA_zp(cb, T32 + 1)
    LDA_zp(cb, T32 + 2); ADC_zp(cb, PROD + 2); STA_zp(cb, T32 + 2)
    LDA_zp(cb, T32 + 3); ADC_zp(cb, PROD + 3); STA_zp(cb, T32 + 3)
    # advance KROW by STRIDE (we're iterating positions)
    CLC(cb)
    LDA_zp(cb, KROW);     ADC_imm(cb, STRIDE); STA_zp(cb, KROW)
    LDA_zp(cb, KROW + 1); ADC_imm(cb, 0);      STA_zp(cb, KROW + 1)
    # ++TIDX
    INC_zp(cb, TIDX)
    LDA_zp(cb, TIDX)
    cb.emit(0xC5, NKEYS)
    cb.emit_branch_far(0x90, '_ah_vt_lp')

    # divide T32 by WSUM, result in SCR_A[0..1]
    cb.emit_jsr('sdiv')
    # store quotient to OP[J*2]
    LDA_zp(cb, J); ASL_A(cb); TAY(cb)
    LDA_zp(cb, SCR_A);     STA_indY(cb, OP)
    INY_op(cb)
    LDA_zp(cb, SCR_A + 1); STA_indY(cb, OP)

    INC_zp(cb, J)
    LDA_zp(cb, J); CMP_imm(cb, HD)
    cb.emit_branch_far(0x90, '_ah_out_lp')

    RTS(cb)


# ─── test me baby ──────────────────────────────────────────────────
def test_attn_head():
    import shadow as sh
    from numerics import EXP_LUT, ED, SL as SL_const, NH

    rng = np.random.default_rng(0)

    code_org = 0x0800
    cb = CodeBuilder(org=code_org)
    build_smul16(cb)
    build_sdiv_i32_u16(cb)
    build_attn_head(cb)
    code_bytes = cb.get_bytes()
    entry = cb.labels['attn_head']
    print(f"attn_head built: {len(code_bytes)} bytes, entry ${entry:04X}")

    # mem layout for the test
    Q_ADDR  = 0x4000   # HD int16 = 16 bytes
    K_ADDR  = 0x4100   # SL positions × STRIDE = 1280 bytes
    V_ADDR  = 0x4700
    O_ADDR  = 0x4D00   # HD int16 = 16 bytes
    SCORES_ADDR = 0x4E00   # SL × 2 = 40 bytes (as int16 sf)
    WTS_ADDR    = 0x4E80   # SL × 1 = 20 bytes
    ELUT_ADDR   = 0x4F00   # 128 bytes

    all_ok = True
    for case_i in range(6):
        r = np.random.default_rng(case_i + 50)
        T = r.integers(2, 12)     # No of keys
        head = int(r.integers(0, NH))
        q = r.integers(-5000, 5001, size=HD, dtype=np.int16)
        # build full K/V matrices for NH heads over T positions. we only use
        # the given head's slice, but the attn_head routine reads from the
        # full K_ALL/V_ALL-style buffer with stride STRIDE
        K_full = r.integers(-5000, 5001, size=(T, ED), dtype=np.int16)
        V_full = r.integers(-5000, 5001, size=(T, ED), dtype=np.int16)

        # da shadow reference
        sh.MEM[Q_ADDR:Q_ADDR + 2*HD] = b'\0' * (2*HD)
        # q is placed at Q_ADDR
        for j, v in enumerate(q):
            sh.store_i16(Q_ADDR + j*2, int(v))
        # K_full, V_full placed as [T][ED] int16 at K_ADDR / V_ADDR, stride=STRIDE
        for t in range(T):
            for d in range(ED):
                sh.store_i16(K_ADDR + t*STRIDE + d*2, int(K_full[t, d]))
                sh.store_i16(V_ADDR + t*STRIDE + d*2, int(V_full[t, d]))
        sh.op_attn_head(Q_ADDR, K_ADDR, V_ADDR, T, head, O_ADDR)
        expected = np.array([sh.load_i16(O_ADDR + j*2) for j in range(HD)],
                            dtype=np.int16)

        # 6502 run, forest, run!
        def setup(cpu, _T=T, _head=head, _q=q, _K=K_full, _V=V_full):
            # zero output
            for i in range(HD * 2):
                cpu.mem[O_ADDR + i] = 0
            # q at Q_ADDR
            for j, v in enumerate(_q):
                vi = int(v) & 0xFFFF
                cpu.mem[Q_ADDR + j*2]     = vi & 0xFF
                cpu.mem[Q_ADDR + j*2 + 1] = (vi >> 8) & 0xFF
            # K, V
            for t in range(_T):
                for d in range(ED):
                    kv = int(_K[t, d]) & 0xFFFF
                    vv = int(_V[t, d]) & 0xFFFF
                    cpu.mem[K_ADDR + t*STRIDE + d*2]     = kv & 0xFF
                    cpu.mem[K_ADDR + t*STRIDE + d*2 + 1] = (kv >> 8) & 0xFF
                    cpu.mem[V_ADDR + t*STRIDE + d*2]     = vv & 0xFF
                    cpu.mem[V_ADDR + t*STRIDE + d*2 + 1] = (vv >> 8) & 0xFF
            # EXP_LUT
            for i in range(128):
                cpu.mem[ELUT_ADDR + i] = int(EXP_LUT[i])
            # p p p p pointers
            cpu.mem[QP]     = Q_ADDR & 0xFF
            cpu.mem[QP + 1] = (Q_ADDR >> 8) & 0xFF
            cpu.mem[KB]     = K_ADDR & 0xFF
            cpu.mem[KB + 1] = (K_ADDR >> 8) & 0xFF
            cpu.mem[VB]     = V_ADDR & 0xFF
            cpu.mem[VB + 1] = (V_ADDR >> 8) & 0xFF
            cpu.mem[OP]     = O_ADDR & 0xFF
            cpu.mem[OP + 1] = (O_ADDR >> 8) & 0xFF
            cpu.mem[NKEYS]  = _T
            cpu.mem[HEAD]   = _head
            cpu.mem[SCORES_P]     = SCORES_ADDR & 0xFF
            cpu.mem[SCORES_P + 1] = (SCORES_ADDR >> 8) & 0xFF
            cpu.mem[WTS_P]        = WTS_ADDR & 0xFF
            cpu.mem[WTS_P + 1]    = (WTS_ADDR >> 8) & 0xFF
            cpu.mem[ELUT_P]       = ELUT_ADDR & 0xFF
            cpu.mem[ELUT_P + 1]   = (ELUT_ADDR >> 8) & 0xFF

        cpu = run_subroutine(code_bytes, code_org, entry, setup,
                             max_cycles=5_000_000)

        got = np.zeros(HD, dtype=np.int16)
        for j in range(HD):
            lo = cpu.mem[O_ADDR + j*2]
            hi = cpu.mem[O_ADDR + j*2 + 1]
            v = lo | (hi << 8)
            if v & 0x8000: v -= 0x10000
            got[j] = v

        match = np.array_equal(got, expected)
        status = "OK" if match else f"FAIL ({int((got != expected).sum())} diffs)"
        print(f"  attn_head case{case_i} T={T} head={head} "
              f"cycles={cpu.cycles:6d}  {status}")
        if not match:
            all_ok = False
            print(f"    got     = {got.tolist()}")
            print(f"    expected= {expected.tolist()}")

    print("\n  OVERALL:", "ALL PASS - ACCES ALL AREAS!" if all_ok else "FAILURES")


if __name__ == '__main__':
    test_attn_head()
