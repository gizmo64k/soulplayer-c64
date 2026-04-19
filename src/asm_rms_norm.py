#!/usr/bin/env python3
"""
6502 rms_norm + parity test.

Algorithm (matches bob_numerics.rms_norm and bob_shadow_v3.op_rms_norm):
    sum_sq = Σ (x[i] >> 4)²                     # int32
    mean_sq = max(1, sum_sq // ED)               # int32
    rms = max(1, isqrt_u32(mean_sq))              # int16
    inv = udiv_u32_u16(1<<20, rms)                # u16
    for i in 0..ED-1:
        y_raw = (x[i] * inv) >> 16                # signed 16×u16 → take mid 16
        dst[i] = sat16((y_raw * gain[i]) >> s_g)

Calling convention (zero page):
  $F0/$F1  XP     x pointer (int16 Q8.8, ED elements)
  $F2/$F3  GP     gain pointer (int8, ED elements)
  $F4/$F5  DP     dst pointer (int16 Q8.8 output, ED elements)
  $F6      SG     gain shift (u8)

Scratch (matches matvec where possible so we can share smul_add):
  $10..$13  SUMSQ  32-bit accumulator for sum of squares
  $14       IDX    loop counter
  $18       TMP    (used by smul_add)
  $19/$1A   SRC16
  $1B       SIGN
  $1C..$1F  PROD
  $20/$21   RMS    isqrt result (u16)
  $22/$23   INV    inverse (u16)
  $24..$27  TMP32  scratch 32-bit (sqrt/divide working space)
"""
import numpy as np
from assembler import (
    CodeBuilder, run_subroutine,
    LDA_imm, LDX_imm, LDY_imm, LDA_zp, LDX_zp, LDY_zp,
    STA_zp, LDA_indY, STA_indY,
    ADC_imm, ADC_zp, SBC_imm, SBC_zp, EOR_imm, CMP_imm,
    INC_zp, DEC_zp, ASL_zp, ROL_zp, ROR_zp, LSR_zp, ASL_A, LSR_A,
    CLC, SEC, RTS, BRK, PHA, PLA, TAY, TYA, TXA, TAX, DEX, INX,
)

# Zero page — non-overlapping blocks.
# Matvec uses $10..$17. We use $18..$3F.
XP, GP, DP = 0x40, 0x42, 0x44
SG = 0x46

SUMSQ = 0x10    # 4 bytes  (aliases matvec's ACC32; safe because they're
                #           never live at the same time)
IDX   = 0x14    # 1 byte

TMP   = 0x18    # 2 bytes: $18/$19  — smul16 multiplier A
SRC16 = 0x1A    # 2 bytes: $1A/$1B  — smul16 multiplier B
SIGN  = 0x1C    # 1 byte            — smul16 sign flag
PROD  = 0x20    # 4 bytes: $20..$23 — smul16 output

RMS   = 0x30    # 2 bytes
INV   = 0x32    # 2 bytes
T32   = 0x34    # 4 bytes: $34..$37 — isqrt bit mask / udiv remainder
SCR_A = 0x38    # 4 bytes: $38..$3B — scratch (isqrt candidate, udiv sub scratch)
SCR_B = 0x3C    # 4 bytes: $3C..$3F — scratch (isqrt subtract scratch)


def INY_op(cb): cb.emit(0xC8)
def DEY_op(cb): cb.emit(0x88)


def build_smul16(cb):
    """Signed 16×16 → 32 multiply.
    Inputs: (TMP/TMP+1) × (SRC16/SRC16+1) signed int16 each
    Output: PROD[0..3] signed int32
    Clobbers A, X, Y, SIGN, and (TMP), (SRC16) (they get absoluted).
    """
    cb.label('smul16')
    LDA_imm(cb, 0); STA_zp(cb, SIGN)

    # Negate TMP (multiplier) if negative
    LDA_zp(cb, TMP + 1)
    cb.emit_branch(0x10, '_sm16_a_pos')   # BPL
    LDA_zp(cb, TMP);     EOR_imm(cb, 0xFF); STA_zp(cb, TMP)
    LDA_zp(cb, TMP + 1); EOR_imm(cb, 0xFF); STA_zp(cb, TMP + 1)
    INC_zp(cb, TMP)
    cb.emit_branch(0xD0, '_sm16_a_nc')    # BNE
    INC_zp(cb, TMP + 1)
    cb.label('_sm16_a_nc')
    LDA_imm(cb, 1); STA_zp(cb, SIGN)
    cb.label('_sm16_a_pos')

    # Negate SRC16 if negative
    LDA_zp(cb, SRC16 + 1)
    cb.emit_branch(0x10, '_sm16_b_pos')
    LDA_zp(cb, SRC16);     EOR_imm(cb, 0xFF); STA_zp(cb, SRC16)
    LDA_zp(cb, SRC16 + 1); EOR_imm(cb, 0xFF); STA_zp(cb, SRC16 + 1)
    INC_zp(cb, SRC16)
    cb.emit_branch(0xD0, '_sm16_b_nc')
    INC_zp(cb, SRC16 + 1)
    cb.label('_sm16_b_nc')
    LDA_zp(cb, SIGN); EOR_imm(cb, 1); STA_zp(cb, SIGN)
    cb.label('_sm16_b_pos')

    # Unsigned 16×16 → 32 multiply via shift-and-add
    # PROD = 0
    LDA_imm(cb, 0)
    STA_zp(cb, PROD);     STA_zp(cb, PROD + 1)
    STA_zp(cb, PROD + 2); STA_zp(cb, PROD + 3)
    LDX_imm(cb, 16)
    cb.label('_sm16_loop')
    # Shift PROD left 1 bit (4-byte shift)
    ASL_zp(cb, PROD)
    ROL_zp(cb, PROD + 1)
    ROL_zp(cb, PROD + 2)
    ROL_zp(cb, PROD + 3)
    # Shift TMP left; bit 15 goes to C
    ASL_zp(cb, TMP)
    ROL_zp(cb, TMP + 1)
    cb.emit_branch(0x90, '_sm16_skip')  # BCC
    # PROD += (SRC16 sign-extended to 32)
    CLC(cb)
    LDA_zp(cb, PROD);     ADC_zp(cb, SRC16);     STA_zp(cb, PROD)
    LDA_zp(cb, PROD + 1); ADC_zp(cb, SRC16 + 1); STA_zp(cb, PROD + 1)
    LDA_zp(cb, PROD + 2); ADC_imm(cb, 0);        STA_zp(cb, PROD + 2)
    LDA_zp(cb, PROD + 3); ADC_imm(cb, 0);        STA_zp(cb, PROD + 3)
    cb.label('_sm16_skip')
    DEX(cb)
    cb.emit_branch_far(0xD0, '_sm16_loop')

    # Negate PROD if sign set
    LDA_zp(cb, SIGN)
    cb.emit_branch(0xF0, '_sm16_done')     # BEQ
    LDA_zp(cb, PROD);     EOR_imm(cb, 0xFF); STA_zp(cb, PROD)
    LDA_zp(cb, PROD + 1); EOR_imm(cb, 0xFF); STA_zp(cb, PROD + 1)
    LDA_zp(cb, PROD + 2); EOR_imm(cb, 0xFF); STA_zp(cb, PROD + 2)
    LDA_zp(cb, PROD + 3); EOR_imm(cb, 0xFF); STA_zp(cb, PROD + 3)
    CLC(cb)
    LDA_zp(cb, PROD);     ADC_imm(cb, 1); STA_zp(cb, PROD)
    LDA_zp(cb, PROD + 1); ADC_imm(cb, 0); STA_zp(cb, PROD + 1)
    LDA_zp(cb, PROD + 2); ADC_imm(cb, 0); STA_zp(cb, PROD + 2)
    LDA_zp(cb, PROD + 3); ADC_imm(cb, 0); STA_zp(cb, PROD + 3)
    cb.label('_sm16_done')
    RTS(cb)


def build_isqrt32(cb):
    """Integer sqrt of a 32-bit unsigned value.
    Input: SUMSQ[0..3] — the value to sqrt (should be > 0)
    Output: RMS[0..1] — floor(sqrt(SUMSQ))
    Clobbers: A, X, Y, T32 (scratch), consumes SUMSQ.

    Bit-by-bit method: result = 0, bit = highest_even_power_of_2 <= n.
    While bit:
        if n >= result + bit:
            n -= result + bit
            result = (result >> 1) + bit
        else:
            result >>= 1
        bit >>= 2

    On 16-bit result, bit starts at 2^14 and shifts down by 2 each iteration,
    so 8 iterations suffice.
    """
    cb.label('isqrt32')
    # result (RMS) = 0
    LDA_imm(cb, 0)
    STA_zp(cb, RMS); STA_zp(cb, RMS + 1)
    # bit = 0x00004000 (bit 14) in T32[0..3]
    STA_zp(cb, T32); STA_zp(cb, T32 + 2); STA_zp(cb, T32 + 3)
    LDA_imm(cb, 0x40); STA_zp(cb, T32 + 1)

    # 8 iterations
    LDX_imm(cb, 8)
    cb.label('_isq_lp')
    # Compute sum = result + bit (into A temp via subtract test):
    # We need to compare n (SUMSQ) to (result_shifted_up + bit). Instead
    # of computing that separately, we can use the standard trick: treat
    # the running "result" as it naturally fits into the high bits of a
    # running sum. Simpler: use explicit 32-bit compare-and-subtract.
    #
    # candidate = result | bit  (since bit is above any set bits of result)
    # Actually no — result grows to touch bit too. Do it literally:
    # Form T32+ RMS expanded as 32-bit sum into $28..$2B.
    CLC(cb)
    LDA_zp(cb, RMS);     ADC_zp(cb, T32);     STA_zp(cb, SCR_A)
    LDA_zp(cb, RMS + 1); ADC_zp(cb, T32 + 1); STA_zp(cb, SCR_A + 1)
    LDA_imm(cb, 0);      ADC_zp(cb, T32 + 2); STA_zp(cb, SCR_A + 2)
    LDA_imm(cb, 0);      ADC_zp(cb, T32 + 3); STA_zp(cb, SCR_A + 3)
    # Compare SUMSQ >= candidate: do 32-bit unsigned compare by subtracting
    # candidate from SUMSQ. If no borrow, SUMSQ >= candidate.
    SEC(cb)
    LDA_zp(cb, SUMSQ);     SBC_zp(cb, SCR_A);     STA_zp(cb, SCR_B)
    LDA_zp(cb, SUMSQ + 1); SBC_zp(cb, SCR_A + 1); STA_zp(cb, SCR_B + 1)
    LDA_zp(cb, SUMSQ + 2); SBC_zp(cb, SCR_A + 2); STA_zp(cb, SCR_B + 2)
    LDA_zp(cb, SUMSQ + 3); SBC_zp(cb, SCR_A + 3); STA_zp(cb, SCR_B + 3)
    cb.emit_branch(0x90, '_isq_less')     # BCC: borrow → SUMSQ < cand
    # SUMSQ >= cand: commit subtraction, set result |= bit
    LDA_zp(cb, SCR_B);     STA_zp(cb, SUMSQ)
    LDA_zp(cb, SCR_B + 1); STA_zp(cb, SUMSQ + 1)
    LDA_zp(cb, SCR_B + 2); STA_zp(cb, SUMSQ + 2)
    LDA_zp(cb, SCR_B + 3); STA_zp(cb, SUMSQ + 3)
    # result = (result >> 1) + bit
    LSR_zp(cb, RMS + 1); ROR_zp(cb, RMS)
    CLC(cb)
    LDA_zp(cb, RMS);     ADC_zp(cb, T32);     STA_zp(cb, RMS)
    LDA_zp(cb, RMS + 1); ADC_zp(cb, T32 + 1); STA_zp(cb, RMS + 1)
    cb.emit_jmp('_isq_next')
    cb.label('_isq_less')
    # result >>= 1
    LSR_zp(cb, RMS + 1); ROR_zp(cb, RMS)
    cb.label('_isq_next')
    # bit >>= 2
    LSR_zp(cb, T32 + 3); ROR_zp(cb, T32 + 2); ROR_zp(cb, T32 + 1); ROR_zp(cb, T32)
    LSR_zp(cb, T32 + 3); ROR_zp(cb, T32 + 2); ROR_zp(cb, T32 + 1); ROR_zp(cb, T32)
    DEX(cb)
    cb.emit_branch_far(0xD0, '_isq_lp')
    RTS(cb)


def build_udiv(cb):
    """
    udiv: compute INV = min(0xFFFF, 0x100000 // RMS) assuming RMS is a
    nonzero u16. Uses restoring division, 16 iterations.

    Working: T32[0..3] holds remainder (starts = 0x100000), quotient built
    into INV[0..1]. We shift T32 left and conditionally subtract RMS.
    """
    cb.label('udiv')
    # T32 = 0x00080000 = 2^19
    LDA_imm(cb, 0); STA_zp(cb, T32); STA_zp(cb, T32 + 1); STA_zp(cb, T32 + 3)
    LDA_imm(cb, 0x08); STA_zp(cb, T32 + 2)
    LDA_imm(cb, 0); STA_zp(cb, INV); STA_zp(cb, INV + 1)

    LDX_imm(cb, 16)
    cb.label('_udiv_lp')
    # Shift T32 left 1, quotient left 1
    ASL_zp(cb, T32); ROL_zp(cb, T32 + 1); ROL_zp(cb, T32 + 2); ROL_zp(cb, T32 + 3)
    ROL_zp(cb, INV); ROL_zp(cb, INV + 1)
    # Try T32[hi] -= RMS (16-bit: T32+2, T32+3 vs RMS, RMS+1)
    SEC(cb)
    LDA_zp(cb, T32 + 2); SBC_zp(cb, RMS);     STA_zp(cb, SCR_A)
    LDA_zp(cb, T32 + 3); SBC_zp(cb, RMS + 1); STA_zp(cb, SCR_A + 1)
    cb.emit_branch(0x90, '_udiv_noss')      # BCC: no subtract
    # Commit subtraction and set quotient bit 0
    LDA_zp(cb, SCR_A);     STA_zp(cb, T32 + 2)
    LDA_zp(cb, SCR_A + 1); STA_zp(cb, T32 + 3)
    LDA_zp(cb, INV); cb.emit(0x09, 0x01); STA_zp(cb, INV)   # ORA #1
    cb.label('_udiv_noss')
    DEX(cb)
    cb.emit_branch_far(0xD0, '_udiv_lp')
    RTS(cb)


def build_rms_norm(cb: CodeBuilder):
    """Assemble the full rms_norm routine."""
    # Helpers first (the main routine JSRs into them)
    build_smul16(cb)
    build_isqrt32(cb)
    build_udiv(cb)

    # ─── Main rms_norm ─────────────────────────────────────────────
    cb.label('rms_norm')

    # Step 1: SUMSQ = Σ (x[i] >> 4)² for i in 0..31
    LDA_imm(cb, 0)
    STA_zp(cb, SUMSQ); STA_zp(cb, SUMSQ + 1)
    STA_zp(cb, SUMSQ + 2); STA_zp(cb, SUMSQ + 3)
    STA_zp(cb, IDX)

    cb.label('_rms_p1')
    # Load x[IDX] into TMP as int16
    LDA_zp(cb, IDX); ASL_A(cb); TAY(cb)
    LDA_indY(cb, XP); STA_zp(cb, TMP)
    INY_op(cb)
    LDA_indY(cb, XP); STA_zp(cb, TMP + 1)
    # Arithmetic shift right 4 (signed): 4× (CMP #$80 ; ROR hi ; ROR lo)
    for _ in range(4):
        LDA_zp(cb, TMP + 1); CMP_imm(cb, 0x80)
        ROR_zp(cb, TMP + 1); ROR_zp(cb, TMP)
    # Square it: copy TMP into SRC16 and call smul16
    LDA_zp(cb, TMP);     STA_zp(cb, SRC16)
    LDA_zp(cb, TMP + 1); STA_zp(cb, SRC16 + 1)
    # Save IDX across smul16 (it clobbers X)
    LDA_zp(cb, IDX); PHA(cb)
    cb.emit_jsr('smul16')
    PLA(cb); STA_zp(cb, IDX)
    # SUMSQ += PROD (32-bit signed add; PROD is positive since it's a square)
    CLC(cb)
    LDA_zp(cb, SUMSQ);     ADC_zp(cb, PROD);     STA_zp(cb, SUMSQ)
    LDA_zp(cb, SUMSQ + 1); ADC_zp(cb, PROD + 1); STA_zp(cb, SUMSQ + 1)
    LDA_zp(cb, SUMSQ + 2); ADC_zp(cb, PROD + 2); STA_zp(cb, SUMSQ + 2)
    LDA_zp(cb, SUMSQ + 3); ADC_zp(cb, PROD + 3); STA_zp(cb, SUMSQ + 3)
    # ++IDX, loop while IDX < 32
    INC_zp(cb, IDX)
    LDA_zp(cb, IDX); CMP_imm(cb, 32)
    cb.emit_branch_far(0x90, '_rms_p1')

    # Step 2: mean_sq = SUMSQ >> 5 (shift right 5 bits logically; SUMSQ is >= 0)
    LDX_imm(cb, 5)
    cb.label('_rms_div32')
    LSR_zp(cb, SUMSQ + 3); ROR_zp(cb, SUMSQ + 2); ROR_zp(cb, SUMSQ + 1); ROR_zp(cb, SUMSQ)
    DEX(cb)
    cb.emit_branch_far(0xD0, '_rms_div32')

    # Ensure SUMSQ >= 1
    LDA_zp(cb, SUMSQ)
    cb.emit(0x05, SUMSQ + 1)      # ORA zp
    cb.emit(0x05, SUMSQ + 2)
    cb.emit(0x05, SUMSQ + 3)
    cb.emit_branch(0xD0, '_rms_sqnz')    # BNE
    LDA_imm(cb, 1); STA_zp(cb, SUMSQ)
    cb.label('_rms_sqnz')

    # Step 3: RMS = isqrt32(SUMSQ)
    cb.emit_jsr('isqrt32')
    # Ensure RMS >= 1
    LDA_zp(cb, RMS); cb.emit(0x05, RMS + 1)  # ORA RMS+1
    cb.emit_branch(0xD0, '_rms_rnz')
    LDA_imm(cb, 1); STA_zp(cb, RMS)
    cb.label('_rms_rnz')

    # Step 4: INV = udiv(0x80000, RMS), capped to 32767
    cb.emit_jsr('udiv')
    # Clamp INV to 0x7FFF if it overflows signed range
    LDA_zp(cb, INV + 1)
    cb.emit_branch(0x10, '_rms_inv_ok')     # BPL: INV < 32768, fine
    LDA_imm(cb, 0xFF); STA_zp(cb, INV)
    LDA_imm(cb, 0x7F); STA_zp(cb, INV + 1)
    cb.label('_rms_inv_ok')

    # Step 5: y[i] = sat16((((x[i] * INV) >> 15) * g[i]) >> SG)
    LDA_imm(cb, 0); STA_zp(cb, IDX)

    cb.label('_rms_p2')
    # TMP = x[IDX]
    LDA_zp(cb, IDX); ASL_A(cb); TAY(cb)
    LDA_indY(cb, XP); STA_zp(cb, TMP)
    INY_op(cb)
    LDA_indY(cb, XP); STA_zp(cb, TMP + 1)
    # SRC16 = INV (now guaranteed ≤ 32767, safe for signed smul16)
    LDA_zp(cb, INV);     STA_zp(cb, SRC16)
    LDA_zp(cb, INV + 1); STA_zp(cb, SRC16 + 1)
    LDA_zp(cb, IDX); PHA(cb)
    cb.emit_jsr('smul16')
    PLA(cb); STA_zp(cb, IDX)
    # y_raw = PROD >> 15: arithmetic right shift 15 on the full 32-bit PROD,
    # then take the low 16 bits (PROD[0..1]). Doing it on 32 bits avoids the
    # overflow case where PROD >> 8 has bit 15 set but PROD is actually positive.
    LDX_imm(cb, 15)
    cb.label('_rms_y_shift')
    LDA_zp(cb, PROD + 3); CMP_imm(cb, 0x80)
    ROR_zp(cb, PROD + 3); ROR_zp(cb, PROD + 2); ROR_zp(cb, PROD + 1); ROR_zp(cb, PROD)
    DEX(cb)
    cb.emit_branch_far(0xD0, '_rms_y_shift')
    # Now PROD[0..1] is the signed y_raw. Copy to TMP.
    LDA_zp(cb, PROD);     STA_zp(cb, TMP)
    LDA_zp(cb, PROD + 1); STA_zp(cb, TMP + 1)
    # Load g[IDX] as int8 and sign-extend to SRC16
    LDY_zp(cb, IDX)
    LDA_indY(cb, GP)
    STA_zp(cb, SRC16)
    # Sign-extend
    cb.emit_branch(0x10, '_rms_g_pos')     # BPL
    LDA_imm(cb, 0xFF); STA_zp(cb, SRC16 + 1)
    cb.emit_jmp('_rms_g_done')
    cb.label('_rms_g_pos')
    LDA_imm(cb, 0x00); STA_zp(cb, SRC16 + 1)
    cb.label('_rms_g_done')
    # Multiply TMP × SRC16
    LDA_zp(cb, IDX); PHA(cb)
    cb.emit_jsr('smul16')
    PLA(cb); STA_zp(cb, IDX)
    # Shift PROD right by SG (arithmetic, 32-bit)
    LDX_zp(cb, SG)
    cb.emit_branch(0xF0, '_rms_no_shift')    # BEQ: skip when SG=0
    cb.label('_rms_shift_lp')
    LDA_zp(cb, PROD + 3); CMP_imm(cb, 0x80)
    ROR_zp(cb, PROD + 3); ROR_zp(cb, PROD + 2); ROR_zp(cb, PROD + 1); ROR_zp(cb, PROD)
    DEX(cb)
    cb.emit_branch_far(0xD0, '_rms_shift_lp')
    cb.label('_rms_no_shift')
    # Saturate PROD[0..1] based on PROD[3] sign
    LDA_zp(cb, PROD + 3)
    cb.emit_branch(0x30, '_rms_sat_neg_chk')    # BMI
    # Positive: PROD[2] and [3] must be 0, PROD[1] bit 7 must be 0
    LDA_zp(cb, PROD + 2)
    cb.emit_branch_far(0xD0, '_rms_sat_pos')
    LDA_zp(cb, PROD + 3)
    cb.emit_branch_far(0xD0, '_rms_sat_pos')
    LDA_zp(cb, PROD + 1)
    cb.emit_branch_far(0x30, '_rms_sat_pos')
    cb.emit_jmp('_rms_sat_store')
    cb.label('_rms_sat_neg_chk')
    LDA_zp(cb, PROD + 2); CMP_imm(cb, 0xFF)
    cb.emit_branch_far(0xD0, '_rms_sat_neg')
    LDA_zp(cb, PROD + 3); CMP_imm(cb, 0xFF)
    cb.emit_branch_far(0xD0, '_rms_sat_neg')
    LDA_zp(cb, PROD + 1)
    cb.emit_branch_far(0x10, '_rms_sat_neg')
    cb.emit_jmp('_rms_sat_store')
    cb.label('_rms_sat_pos')
    LDA_imm(cb, 0xFF); STA_zp(cb, PROD)
    LDA_imm(cb, 0x7F); STA_zp(cb, PROD + 1)
    cb.emit_jmp('_rms_sat_store')
    cb.label('_rms_sat_neg')
    LDA_imm(cb, 0x00); STA_zp(cb, PROD)
    LDA_imm(cb, 0x80); STA_zp(cb, PROD + 1)
    cb.label('_rms_sat_store')
    # Store PROD[0..1] to dst[IDX]
    LDA_zp(cb, IDX); ASL_A(cb); TAY(cb)
    LDA_zp(cb, PROD);     STA_indY(cb, DP)
    INY_op(cb)
    LDA_zp(cb, PROD + 1); STA_indY(cb, DP)
    # ++IDX, loop while IDX < 32
    INC_zp(cb, IDX)
    LDA_zp(cb, IDX); CMP_imm(cb, 32)
    cb.emit_branch_far(0x90, '_rms_p2')
    RTS(cb)


# ─── Test harness ──────────────────────────────────────────────────
def test_rms_norm():
    import shadow as sh
    from numerics import ED, pack_tensor

    rng = np.random.default_rng(0)

    code_org = 0x0900
    cb = CodeBuilder(org=code_org)
    build_rms_norm(cb)
    code_bytes = cb.get_bytes()
    entry = cb.labels['rms_norm']
    print(f"rms_norm built: {len(code_bytes)} bytes, entry ${entry:04X}")

    # Several test cases with different input magnitudes
    cases = []
    for seed in range(6):
        r = np.random.default_rng(seed + 100)
        x = r.integers(-5000, 5001, size=ED, dtype=np.int16)
        g_float = r.normal(1.0, 0.2, size=ED)
        gain = pack_tensor(g_float.astype(np.float32))
        cases.append((f"seed{seed}", x, gain))

    X_ADDR = 0x5000
    G_ADDR = 0x5100
    D_ADDR = 0x5200

    all_ok = True
    for name, x, gain in cases:
        # Shadow reference
        for i, v in enumerate(x):
            sh.store_i16(X_ADDR + i * 2, int(v))
        for i, v in enumerate(gain['q']):
            sh.MEM[G_ADDR + i] = int(v) & 0xFF
        for i in range(ED * 2):
            sh.MEM[D_ADDR + i] = 0
        sh.op_rms_norm(X_ADDR, G_ADDR, gain['s'], D_ADDR)
        expected = np.array(
            [sh.load_i16(D_ADDR + i * 2) for i in range(ED)], dtype=np.int16)

        def setup(cpu):
            for i, v in enumerate(x):
                vi = int(v) & 0xFFFF
                cpu.mem[X_ADDR + i * 2]     = vi & 0xFF
                cpu.mem[X_ADDR + i * 2 + 1] = (vi >> 8) & 0xFF
            for i, v in enumerate(gain['q']):
                cpu.mem[G_ADDR + i] = int(v) & 0xFF
            for i in range(ED * 2):
                cpu.mem[D_ADDR + i] = 0
            cpu.mem[XP]     = X_ADDR & 0xFF
            cpu.mem[XP + 1] = (X_ADDR >> 8) & 0xFF
            cpu.mem[GP]     = G_ADDR & 0xFF
            cpu.mem[GP + 1] = (G_ADDR >> 8) & 0xFF
            cpu.mem[DP]     = D_ADDR & 0xFF
            cpu.mem[DP + 1] = (D_ADDR >> 8) & 0xFF
            cpu.mem[SG]     = gain['s']

        cpu = run_subroutine(code_bytes, code_org, entry, setup,
                             max_cycles=2_000_000)

        got = np.zeros(ED, dtype=np.int16)
        for i in range(ED):
            lo = cpu.mem[D_ADDR + i * 2]
            hi = cpu.mem[D_ADDR + i * 2 + 1]
            v = lo | (hi << 8)
            if v & 0x8000: v -= 0x10000
            got[i] = v

        match = np.array_equal(got, expected)
        diffs = int((got != expected).sum())
        status = "OK" if match else f"FAIL ({diffs} diffs)"
        print(f"  rms_norm {name} s_g={gain['s']} cycles={cpu.cycles:6d}  {status}")
        if not match:
            all_ok = False
            idx = int(np.argmax(np.abs(
                got.astype(np.int32) - expected.astype(np.int32))))
            print(f"    worst idx {idx}: got={got[idx]} expected={expected[idx]}")
            print(f"    got[:8]  = {got[:8].tolist()}")
            print(f"    expect[:8]={expected[:8].tolist()}")
            print(f"    cpu RMS={cpu.mem[RMS] | (cpu.mem[RMS+1]<<8)} "
                  f"INV={cpu.mem[INV] | (cpu.mem[INV+1]<<8)}")
            # Also show expected intermediates
            from numerics import isqrt_u32, udiv_u32_u16
            xs = (x.astype(np.int32) >> 4)
            sum_sq = int((xs * xs).sum())
            mean_sq = max(1, sum_sq // ED)
            exp_rms = max(1, isqrt_u32(mean_sq))
            exp_inv = udiv_u32_u16(1 << 20, exp_rms)
            print(f"    expected RMS={exp_rms} INV={exp_inv}")

    print("\n  OVERALL:", "ALL PASS" if all_ok else "FAILURES")


if __name__ == '__main__':
    test_rms_norm()
