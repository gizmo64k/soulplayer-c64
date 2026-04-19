#!/usr/bin/env python3
"""
6502 matvec with labeled branches (no hand-counted distances).

dst[r] = sat16((Σ W[r,c] * src[c]) >> total_shift)

Zero-page calling convention:
  $F0/$F1  WP     W pointer (int8 matrix, row-major)
  $F2/$F3  SP     src pointer (int16 Q8.8 vector)
  $F4/$F5  DP     dst pointer (int16 Q8.8 vector, output)
  $F6      ROWS
  $F7      COLS
  $F8      SHIFT  (s_w + post_shift)

Internal scratch:
  $10..$13  ACC32
  $14       RI
  $15       CI
  $16/$17   WROW (pointer to current row)
  $18       TMP (signed weight)
  $19/$1A   SRC16 (copy of src[c] as 16 bits)
  $1B       SIGN
  $1C..$1F  PROD (32-bit product)
"""
import numpy as np
from assembler import (
    CodeBuilder, run_subroutine,
    LDA_imm, LDX_imm, LDY_imm, LDA_zp, LDX_zp, LDY_zp,
    STA_zp, LDA_indY, STA_indY,
    ADC_imm, ADC_zp, EOR_imm, CMP_imm, INC_zp, DEC_zp,
    ASL_zp, ROL_zp, ROR_zp, ASL_A,
    CLC, RTS, BRK, PHA, PLA, TAY, TYA, DEX,
)

WP, SP, DP = 0x40, 0x42, 0x44
ROWS, COLS, SHIFT = 0x46, 0x47, 0x48
BP = 0x49                   # bias pointer (2 bytes: 0xF9/0xFA)
BFLAG = 0x4B                # 0 = no bias, 1 = bias
ACC32 = 0x10
RI, CI = 0x14, 0x15
WROW = 0x16
TMP = 0x18
SRC16 = 0x19
SIGN = 0x1B
PROD = 0x1C


def build_matvec(cb: CodeBuilder):
    """Append matvec + smul_add helper to cb."""

    # ─── Helper: smul_add ──────────────────────────────────────────
    # Precondition: A = int8 weight, Y = c*2 (byte offset into src)
    # Effect: ACC32 += sign_extend(weight) * src[c]  (32-bit signed add)
    # Preserves: nothing (caller saves X/Y as needed).
    cb.label('smul_add')
    STA_zp(cb, TMP)
    LDA_indY(cb, SP); STA_zp(cb, SRC16)       # SRC16 lo
    INY_op(cb)
    LDA_indY(cb, SP); STA_zp(cb, SRC16 + 1)   # SRC16 hi
    DEY_op(cb)

    LDA_imm(cb, 0); STA_zp(cb, SIGN)

    # If weight negative: negate and flip sign
    LDA_zp(cb, TMP)
    cb.emit_branch(0x10, '_sma_w_pos')        # BPL
    EOR_imm(cb, 0xFF); CLC(cb); ADC_imm(cb, 1); STA_zp(cb, TMP)
    LDA_imm(cb, 1); STA_zp(cb, SIGN)
    cb.label('_sma_w_pos')

    # If src negative: negate and flip sign
    LDA_zp(cb, SRC16 + 1)
    cb.emit_branch(0x10, '_sma_s_pos')        # BPL
    LDA_zp(cb, SRC16);     EOR_imm(cb, 0xFF); STA_zp(cb, SRC16)
    LDA_zp(cb, SRC16 + 1); EOR_imm(cb, 0xFF); STA_zp(cb, SRC16 + 1)
    INC_zp(cb, SRC16)
    cb.emit_branch(0xD0, '_sma_s_no_carry')   # BNE
    INC_zp(cb, SRC16 + 1)
    cb.label('_sma_s_no_carry')
    LDA_zp(cb, SIGN); EOR_imm(cb, 1); STA_zp(cb, SIGN)
    cb.label('_sma_s_pos')

    # Unsigned 8×16 multiply: TMP × (SRC16 hi:lo) → PROD[0..2]
    LDA_imm(cb, 0)
    STA_zp(cb, PROD); STA_zp(cb, PROD + 1); STA_zp(cb, PROD + 2)
    LDX_imm(cb, 8)
    cb.label('_sma_loop')
    ASL_zp(cb, PROD); ROL_zp(cb, PROD + 1); ROL_zp(cb, PROD + 2)
    ASL_zp(cb, TMP)
    cb.emit_branch(0x90, '_sma_skip_add')     # BCC
    CLC(cb)
    LDA_zp(cb, PROD);     ADC_zp(cb, SRC16);     STA_zp(cb, PROD)
    LDA_zp(cb, PROD + 1); ADC_zp(cb, SRC16 + 1); STA_zp(cb, PROD + 1)
    LDA_zp(cb, PROD + 2); ADC_imm(cb, 0);        STA_zp(cb, PROD + 2)
    cb.label('_sma_skip_add')
    DEX(cb)
    cb.emit_branch_far(0xD0, '_sma_loop')     # BNE

    # Sign-extend PROD to 32 bits (PROD[3]), negate if sign set
    LDA_imm(cb, 0); STA_zp(cb, PROD + 3)
    LDA_zp(cb, SIGN)
    cb.emit_branch(0xF0, '_sma_no_negate')    # BEQ
    LDA_zp(cb, PROD);     EOR_imm(cb, 0xFF); STA_zp(cb, PROD)
    LDA_zp(cb, PROD + 1); EOR_imm(cb, 0xFF); STA_zp(cb, PROD + 1)
    LDA_zp(cb, PROD + 2); EOR_imm(cb, 0xFF); STA_zp(cb, PROD + 2)
    LDA_imm(cb, 0xFF); STA_zp(cb, PROD + 3)
    CLC(cb)
    LDA_zp(cb, PROD);     ADC_imm(cb, 1); STA_zp(cb, PROD)
    LDA_zp(cb, PROD + 1); ADC_imm(cb, 0); STA_zp(cb, PROD + 1)
    LDA_zp(cb, PROD + 2); ADC_imm(cb, 0); STA_zp(cb, PROD + 2)
    LDA_zp(cb, PROD + 3); ADC_imm(cb, 0); STA_zp(cb, PROD + 3)
    cb.label('_sma_no_negate')

    # Add PROD (32-bit) into ACC32
    CLC(cb)
    LDA_zp(cb, ACC32);     ADC_zp(cb, PROD);     STA_zp(cb, ACC32)
    LDA_zp(cb, ACC32 + 1); ADC_zp(cb, PROD + 1); STA_zp(cb, ACC32 + 1)
    LDA_zp(cb, ACC32 + 2); ADC_zp(cb, PROD + 2); STA_zp(cb, ACC32 + 2)
    LDA_zp(cb, ACC32 + 3); ADC_zp(cb, PROD + 3); STA_zp(cb, ACC32 + 3)
    RTS(cb)

    # ─── matvec_bias entry ─────────────────────────────────────────
    # Same calling convention as matvec, plus:
    #   $F9/$FA  BP      bias pointer (int16 array, 2 bytes per entry)
    # Sets BFLAG=1 and falls through.
    cb.label('matvec_bias')
    LDA_imm(cb, 1); STA_zp(cb, BFLAG)
    cb.emit_jmp('_mv_init')

    # ─── matvec entry ──────────────────────────────────────────────
    cb.label('matvec')
    LDA_imm(cb, 0); STA_zp(cb, BFLAG)

    cb.label('_mv_init')
    LDA_zp(cb, WP);     STA_zp(cb, WROW)
    LDA_zp(cb, WP + 1); STA_zp(cb, WROW + 1)
    LDA_imm(cb, 0); STA_zp(cb, RI)

    cb.label('_mv_row')
    # Initialize ACC32.
    # If BFLAG is set, seed ACC32 with sign-extended bias[RI] and zero the
    # top two bytes as appropriate. Otherwise zero ACC32 entirely.
    LDA_zp(cb, BFLAG)
    cb.emit_branch(0xF0, '_mv_zero_acc')     # BEQ: no bias → zero
    # Load bias[RI] as int16: Y = RI*2
    LDA_zp(cb, RI); ASL_A(cb); TAY(cb)
    LDA_indY(cb, BP)
    STA_zp(cb, ACC32)
    INY_op(cb)
    LDA_indY(cb, BP)
    STA_zp(cb, ACC32 + 1)
    # Sign-extend into ACC32[2..3]
    LDA_imm(cb, 0)
    LDX_zp(cb, ACC32 + 1)          # load for BMI test; doesn't set flags
    # Use CMP #$80 on ACC32+1 via LDA then CMP to set carry = sign
    LDA_zp(cb, ACC32 + 1)
    cb.emit_branch(0x10, '_mv_bias_pos')     # BPL: positive → high bytes = 0
    LDA_imm(cb, 0xFF)
    STA_zp(cb, ACC32 + 2); STA_zp(cb, ACC32 + 3)
    cb.emit_jmp('_mv_acc_done')
    cb.label('_mv_bias_pos')
    LDA_imm(cb, 0x00)
    STA_zp(cb, ACC32 + 2); STA_zp(cb, ACC32 + 3)
    cb.emit_jmp('_mv_acc_done')

    cb.label('_mv_zero_acc')
    LDA_imm(cb, 0)
    STA_zp(cb, ACC32); STA_zp(cb, ACC32 + 1)
    STA_zp(cb, ACC32 + 2); STA_zp(cb, ACC32 + 3)

    cb.label('_mv_acc_done')
    LDA_imm(cb, 0); STA_zp(cb, CI)

    cb.label('_mv_col')
    LDY_zp(cb, CI)
    LDA_indY(cb, WROW)       # A = W[row, CI]
    # Save A, compute Y = CI*2, restore A
    PHA(cb)
    LDA_zp(cb, CI); ASL_A(cb); TAY(cb)
    PLA(cb)
    cb.emit_jsr('smul_add')
    INC_zp(cb, CI)
    LDA_zp(cb, CI)
    cb.emit(0xC5, COLS)      # CMP zp COLS
    cb.emit_branch_far(0x90, '_mv_col')

    # Shift ACC32 right by SHIFT (arithmetic)
    LDX_zp(cb, SHIFT)
    cb.emit_branch(0xF0, '_mv_no_shift')      # BEQ: skip when shift=0
    cb.label('_mv_shift_lp')
    LDA_zp(cb, ACC32 + 3); CMP_imm(cb, 0x80)  # C = sign bit
    ROR_zp(cb, ACC32 + 3)
    ROR_zp(cb, ACC32 + 2)
    ROR_zp(cb, ACC32 + 1)
    ROR_zp(cb, ACC32)
    DEX(cb)
    cb.emit_branch_far(0xD0, '_mv_shift_lp')
    cb.label('_mv_no_shift')

    # Saturate ACC32[0..1] to int16.
    # Dispatch on the sign of the full 32-bit value (ACC32[3] bit 7).
    LDA_zp(cb, ACC32 + 3)
    cb.emit_branch(0x30, '_mv_neg_check')     # BMI: negative → _mv_neg_check
    # Positive path: ACC32[2] and ACC32[3] must both be 0, AND ACC32[1] bit 7
    # must be 0 (otherwise the low 16 bits look negative to int16).
    LDA_zp(cb, ACC32 + 2)
    cb.emit_branch_far(0xD0, '_mv_sat_pos')
    LDA_zp(cb, ACC32 + 3)
    cb.emit_branch_far(0xD0, '_mv_sat_pos')
    LDA_zp(cb, ACC32 + 1)
    cb.emit_branch_far(0x30, '_mv_sat_pos')   # BMI: would read as negative
    cb.emit_jmp('_mv_store')
    cb.label('_mv_neg_check')
    # Negative path: ACC32[2] and [3] must both be $FF, AND ACC32[1] bit 7
    # must be 1 (otherwise the low 16 bits look positive).
    LDA_zp(cb, ACC32 + 2); CMP_imm(cb, 0xFF)
    cb.emit_branch_far(0xD0, '_mv_sat_neg')
    LDA_zp(cb, ACC32 + 3); CMP_imm(cb, 0xFF)
    cb.emit_branch_far(0xD0, '_mv_sat_neg')
    LDA_zp(cb, ACC32 + 1)
    cb.emit_branch_far(0x10, '_mv_sat_neg')   # BPL: would read as positive
    cb.emit_jmp('_mv_store')

    cb.label('_mv_sat_pos')
    LDA_imm(cb, 0xFF); STA_zp(cb, ACC32)
    LDA_imm(cb, 0x7F); STA_zp(cb, ACC32 + 1)
    cb.emit_jmp('_mv_store')

    cb.label('_mv_sat_neg')
    LDA_imm(cb, 0x00); STA_zp(cb, ACC32)
    LDA_imm(cb, 0x80); STA_zp(cb, ACC32 + 1)

    cb.label('_mv_store')
    LDA_zp(cb, RI); ASL_A(cb); TAY(cb)
    LDA_zp(cb, ACC32);     STA_indY(cb, DP)
    INY_op(cb)
    LDA_zp(cb, ACC32 + 1); STA_indY(cb, DP)

    # Advance WROW by COLS
    CLC(cb)
    LDA_zp(cb, WROW);     ADC_zp(cb, COLS); STA_zp(cb, WROW)
    LDA_zp(cb, WROW + 1); ADC_imm(cb, 0);   STA_zp(cb, WROW + 1)

    INC_zp(cb, RI)
    LDA_zp(cb, RI)
    cb.emit(0xC5, ROWS)      # CMP zp ROWS
    cb.emit_branch_far(0x90, '_mv_row')
    RTS(cb)


# INY/DEY shortcuts (asm_harness has them but they need parens-less wrappers)
def INY_op(cb): cb.emit(0xC8)
def DEY_op(cb): cb.emit(0x88)


# ─── Test harness ──────────────────────────────────────────────────
def test_matvec():
    from shadow import (
        op_matvec, store_i8 as sh_store_i8,
        store_i16 as sh_store_i16, load_i16 as sh_load_i16,
    )

    rng = np.random.default_rng(0)
    cases = [
        (8,  8,  4, 0),
        (8,  8,  6, 1),
        (32, 32, 7, 1),
        (64, 32, 7, 1),
        (32, 64, 7, 1),
        (16, 16, 5, 1),
        (32, 32, 0, 0),
    ]

    code_org = 0x0900
    cb = CodeBuilder(org=code_org)
    build_matvec(cb)
    code_bytes = cb.get_bytes()
    entry = cb.labels['matvec']
    print(f"matvec built: {len(code_bytes)} bytes, entry ${entry:04X}")

    all_ok = True
    for (rows, cols, s_w, post_shift) in cases:
        W = rng.integers(-100, 101, size=(rows, cols), dtype=np.int8)
        x = rng.integers(-20000, 20001, size=cols, dtype=np.int16)

        # Shadow reference
        import shadow as sh
        W_ADDR = 0x4000
        S_ADDR = 0x5000
        D_ADDR = 0x5100
        for i, v in enumerate(W.flatten()):
            sh.MEM[W_ADDR + i] = int(v) & 0xFF
        for i, v in enumerate(x):
            sh.store_i16(S_ADDR + i * 2, int(v))
        for i in range(rows * 2):
            sh.MEM[D_ADDR + i] = 0
        op_matvec(W_ADDR, rows, cols, s_w, S_ADDR, D_ADDR, post_shift=post_shift)
        expected = np.array(
            [sh.load_i16(D_ADDR + i * 2) for i in range(rows)], dtype=np.int16)

        # 6502 run
        def setup(cpu):
            for i, v in enumerate(W.flatten()):
                cpu.mem[W_ADDR + i] = int(v) & 0xFF
            for i, v in enumerate(x):
                vi = int(v) & 0xFFFF
                cpu.mem[S_ADDR + i * 2]     = vi & 0xFF
                cpu.mem[S_ADDR + i * 2 + 1] = (vi >> 8) & 0xFF
            for i in range(rows * 2):
                cpu.mem[D_ADDR + i] = 0
            cpu.mem[WP]     = W_ADDR & 0xFF
            cpu.mem[WP + 1] = (W_ADDR >> 8) & 0xFF
            cpu.mem[SP]     = S_ADDR & 0xFF
            cpu.mem[SP + 1] = (S_ADDR >> 8) & 0xFF
            cpu.mem[DP]     = D_ADDR & 0xFF
            cpu.mem[DP + 1] = (D_ADDR >> 8) & 0xFF
            cpu.mem[ROWS]   = rows
            cpu.mem[COLS]   = cols
            cpu.mem[SHIFT]  = s_w + post_shift

        cpu = run_subroutine(code_bytes, code_org, entry, setup)

        got = np.zeros(rows, dtype=np.int16)
        for i in range(rows):
            lo = cpu.mem[D_ADDR + i * 2]
            hi = cpu.mem[D_ADDR + i * 2 + 1]
            v = lo | (hi << 8)
            if v & 0x8000: v -= 0x10000
            got[i] = v

        match = np.array_equal(got, expected)
        status = "OK" if match else f"FAIL ({int((got != expected).sum())} diffs)"
        print(f"  matvec({rows:2d}x{cols:2d}, s={s_w}, post={post_shift}) "
              f"cycles={cpu.cycles:8d}  {status}")
        if not match:
            all_ok = False
            idx = int(np.argmax(np.abs(
                got.astype(np.int32) - expected.astype(np.int32))))
            print(f"    worst row {idx}: got={got[idx]}, expected={expected[idx]}")
            print(f"    got[:8]  = {got[:8].tolist()}")
            print(f"    expect[:8]={expected[:8].tolist()}")

    # ── matvec_bias tests ──
    from shadow import op_matvec_bias
    entry_bias = cb.labels['matvec_bias']
    print(f"\nmatvec_bias entry ${entry_bias:04X}")

    bias_cases = [
        # (rows, cols, s_w, post_shift)
        (64, 32, 7, 1),   # fc1
        (32, 64, 7, 1),   # fc2
        (32, 32, 6, 1),   # smaller, bias dominates more
        (8,  8,  4, 0),   # no post shift
    ]

    for (rows, cols, s_w, post_shift) in bias_cases:
        W = rng.integers(-100, 101, size=(rows, cols), dtype=np.int8)
        x = rng.integers(-20000, 20001, size=cols, dtype=np.int16)
        b = rng.integers(-5000, 5001, size=rows, dtype=np.int16)

        import shadow as sh
        W_ADDR = 0x4000
        S_ADDR = 0x5000
        D_ADDR = 0x5100
        B_ADDR = 0x5300
        for i, v in enumerate(W.flatten()):
            sh.MEM[W_ADDR + i] = int(v) & 0xFF
        for i, v in enumerate(x):
            sh.store_i16(S_ADDR + i * 2, int(v))
        for i, v in enumerate(b):
            sh.store_i16(B_ADDR + i * 2, int(v))
        for i in range(rows * 2):
            sh.MEM[D_ADDR + i] = 0
        op_matvec_bias(W_ADDR, B_ADDR, rows, cols, s_w, S_ADDR, D_ADDR,
                       post_shift=post_shift)
        expected = np.array(
            [sh.load_i16(D_ADDR + i * 2) for i in range(rows)], dtype=np.int16)

        def setup(cpu):
            for i, v in enumerate(W.flatten()):
                cpu.mem[W_ADDR + i] = int(v) & 0xFF
            for i, v in enumerate(x):
                vi = int(v) & 0xFFFF
                cpu.mem[S_ADDR + i * 2]     = vi & 0xFF
                cpu.mem[S_ADDR + i * 2 + 1] = (vi >> 8) & 0xFF
            for i, v in enumerate(b):
                vi = int(v) & 0xFFFF
                cpu.mem[B_ADDR + i * 2]     = vi & 0xFF
                cpu.mem[B_ADDR + i * 2 + 1] = (vi >> 8) & 0xFF
            for i in range(rows * 2):
                cpu.mem[D_ADDR + i] = 0
            cpu.mem[WP]     = W_ADDR & 0xFF
            cpu.mem[WP + 1] = (W_ADDR >> 8) & 0xFF
            cpu.mem[SP]     = S_ADDR & 0xFF
            cpu.mem[SP + 1] = (S_ADDR >> 8) & 0xFF
            cpu.mem[DP]     = D_ADDR & 0xFF
            cpu.mem[DP + 1] = (D_ADDR >> 8) & 0xFF
            cpu.mem[BP]     = B_ADDR & 0xFF
            cpu.mem[BP + 1] = (B_ADDR >> 8) & 0xFF
            cpu.mem[ROWS]   = rows
            cpu.mem[COLS]   = cols
            cpu.mem[SHIFT]  = s_w + post_shift

        cpu = run_subroutine(code_bytes, code_org, entry_bias, setup)
        got = np.zeros(rows, dtype=np.int16)
        for i in range(rows):
            lo = cpu.mem[D_ADDR + i * 2]
            hi = cpu.mem[D_ADDR + i * 2 + 1]
            v = lo | (hi << 8)
            if v & 0x8000: v -= 0x10000
            got[i] = v

        match = np.array_equal(got, expected)
        status = "OK" if match else f"FAIL ({int((got != expected).sum())} diffs)"
        print(f"  matvec_bias({rows:2d}x{cols:2d}, s={s_w}, post={post_shift}) "
              f"cycles={cpu.cycles:8d}  {status}")
        if not match:
            all_ok = False
            idx = int(np.argmax(np.abs(
                got.astype(np.int32) - expected.astype(np.int32))))
            print(f"    worst row {idx}: got={got[idx]}, expected={expected[idx]}")
            print(f"    got[:8]  = {got[:8].tolist()}")
            print(f"    expect[:8]={expected[:8].tolist()}")

    print("\n  OVERALL:", "ALL PASS" if all_ok else "FAILURES")


if __name__ == '__main__':
    test_matvec()
