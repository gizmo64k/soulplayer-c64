#!/usr/bin/env python3
"""
6502-shaped Python shadow of bob_numerics v3.

Same algorithm, written as a sequence of ops that each mirror a 6502
subroutine exactly: memory-based, integer only, no numpy broadcasting.

If this matches bob_numerics.forward bit-for-bit, the 6502 port has a
clear specification to follow.
"""
import numpy as np
import math
from numerics import (
    Weights, VS, ED, NH, HD, FF, NL, SL, ACT_SHIFT,
    sat16, sat16_arr,
)

# Memory map (same rough layout as planned for the C64)
WEIGHTS_BASE = 0x4000
BUF_BASE     = 0x2800

# Per-position stride: ED int16 entries = 64 bytes
STRIDE = ED * 2

# Activation buffers — allocate SL positions' worth each
HIDDEN  = BUF_BASE + 0x0000       # SL * STRIDE = 1280 bytes = 0x500
Q_ALL   = BUF_BASE + 0x0600
K_ALL   = BUF_BASE + 0x0C00
V_ALL   = BUF_BASE + 0x1200
ATT_OUT = BUF_BASE + 0x1800
SCRATCH = BUF_BASE + 0x1E00       # misc per-op scratch, up to 512 bytes

MEM = bytearray(0x10000)


# ─── Memory primitives ─────────────────────────────────────────────
def store_i8(addr, v): MEM[addr] = v & 0xFF
def load_i8(addr):
    v = MEM[addr]
    return v - 256 if v >= 128 else v

def store_i16(addr, v):
    v = int(v) & 0xFFFF
    MEM[addr] = v & 0xFF
    MEM[addr + 1] = (v >> 8) & 0xFF

def load_i16(addr):
    v = MEM[addr] | (MEM[addr + 1] << 8)
    return v - 0x10000 if v >= 0x8000 else v

def store_i32(addr, v):
    v = int(v) & 0xFFFFFFFF
    for i in range(4):
        MEM[addr + i] = (v >> (8 * i)) & 0xFF

def load_i32(addr):
    v = 0
    for i in range(4):
        v |= MEM[addr + i] << (8 * i)
    return v - 0x100000000 if v >= 0x80000000 else v


# ─── Weight layout in memory ───────────────────────────────────────
class WAddrs:
    te = None
    pe = None
    layers = None
    norm_w = None
    out_w = None


def lay_weights(W: Weights):
    """
    Serialize the Weights object into MEM starting at WEIGHTS_BASE. Returns
    a parallel address map.

    Layout per tensor:
      - int8 weights contiguous in memory
      - shift stored separately (not in memory; the 6502 has it baked into code)
      - int32 biases stored as 4 bytes per element, little-endian
    """
    cur = [WEIGHTS_BASE]
    addrs = {}
    shifts = {}

    def put_i8(name, arr, shift):
        a = cur[0]
        flat = arr.flatten()
        for i, v in enumerate(flat):
            store_i8(a + i, int(v))
        cur[0] += len(flat)
        addrs[name] = a
        shifts[name] = shift

    def put_i16(name, arr, shift):
        a = cur[0]
        flat = arr.flatten()
        for i, v in enumerate(flat):
            store_i16(a + i * 2, int(v))
        cur[0] += len(flat) * 2
        addrs[name] = a
        shifts[name] = shift

    put_i8('te', W.te['q'], W.te['s'])
    put_i8('pe', W.pe['q'], W.pe['s'])
    for L in range(NL):
        lay = W.layers[L]
        put_i8(f'l{L}.n1',    lay['n1']['q'],    lay['n1']['s'])
        put_i8(f'l{L}.q',     lay['q']['q'],     lay['q']['s'])
        put_i8(f'l{L}.k',     lay['k']['q'],     lay['k']['s'])
        put_i8(f'l{L}.v',     lay['v']['q'],     lay['v']['s'])
        put_i8(f'l{L}.proj',  lay['proj']['q'],  lay['proj']['s'])
        put_i8(f'l{L}.n2',    lay['n2']['q'],    lay['n2']['s'])
        put_i8(f'l{L}.fc1_w', lay['fc1_w']['q'], lay['fc1_w']['s'])
        put_i16(f'l{L}.fc1_b', lay['fc1_b']['q16'], lay['fc1_b']['s'])
        put_i8(f'l{L}.fc2_w', lay['fc2_w']['q'], lay['fc2_w']['s'])
        put_i16(f'l{L}.fc2_b', lay['fc2_b']['q16'], lay['fc2_b']['s'])
    put_i8('norm', W.norm_w['q'], W.norm_w['s'])
    put_i8('out',  W.out_w['q'],  W.out_w['s'])

    return addrs, shifts, cur[0] - WEIGHTS_BASE


# ─── Primitive operations (each maps to one 6502 subroutine) ──────

def op_deshift_add_to_hidden(t, tok, te_addr, s_te, pe_addr, s_pe):
    """
    h[t][d] = deshift(te[tok][d], s_te) + deshift(pe[t][d], s_pe)
    where deshift(v, s) shifts v to Q8.8: (v << (8-s)) if 8>=s else (v >> (s-8)).

    On 6502: fixed per-tensor diff -> compile-time shifts.
    """
    h_addr = HIDDEN + t * STRIDE
    diff_te = ACT_SHIFT - s_te
    diff_pe = ACT_SHIFT - s_pe
    for d in range(ED):
        te_v = load_i8(te_addr + tok * ED + d)
        pe_v = load_i8(pe_addr + t * ED + d)
        v1 = (te_v << diff_te) if diff_te >= 0 else (te_v >> -diff_te)
        v2 = (pe_v << diff_pe) if diff_pe >= 0 else (pe_v >> -diff_pe)
        store_i16(h_addr + d * 2, sat16(v1 + v2))


def op_matvec(w_addr, rows, cols, s_w, src_addr, dst_addr, post_shift=1):
    """
    dst[r] = sat16((Σ W[r,c] * src[c]) >> (s_w + post_shift))
    W: int8 row-major at w_addr
    src: int16 Q8.8 at src_addr
    dst: int16 Q8.8 at dst_addr
    """
    total_shift = s_w + post_shift
    for r in range(rows):
        acc = 0  # int32
        for c in range(cols):
            w = load_i8(w_addr + r * cols + c)
            x = load_i16(src_addr + c * 2)
            acc += w * x
        out = acc >> total_shift if total_shift >= 0 else acc << -total_shift
        store_i16(dst_addr + r * 2, sat16(out))


def op_matvec_bias(w_addr, b_addr, rows, cols, s_w, src_addr, dst_addr,
                   post_shift=1):
    """Same as op_matvec but adds a prescaled int16 bias into the accumulator."""
    total_shift = s_w + post_shift
    for r in range(rows):
        acc = 0
        for c in range(cols):
            w = load_i8(w_addr + r * cols + c)
            x = load_i16(src_addr + c * 2)
            acc += w * x
        acc += load_i16(b_addr + r * 2)
        out = acc >> total_shift if total_shift >= 0 else acc << -total_shift
        store_i16(dst_addr + r * 2, sat16(out))


def op_rms_norm(x_addr, g_addr, s_g, dst_addr):
    """
    RMSNorm — integer sqrt + integer divide, matching what the 6502 does.
    """
    from numerics import isqrt_u32, udiv_u32_u16
    sum_sq = 0
    for i in range(ED):
        xi = load_i16(x_addr + i * 2) >> 4
        sum_sq += xi * xi
    mean_sq = max(1, sum_sq // ED)
    rms_shifted = max(1, isqrt_u32(mean_sq))
    inv = udiv_u32_u16(1 << 19, rms_shifted)
    if inv > 32767:
        inv = 32767
    for i in range(ED):
        x = load_i16(x_addr + i * 2)
        y_raw = (x * inv) >> 15
        g = load_i8(g_addr + i)
        y = (y_raw * g) >> s_g
        store_i16(dst_addr + i * 2, sat16(y))


def op_attn_head(q_addr, k_addr_base, v_addr_base, n_keys, head, out_addr):
    """
    Single attention head for one query position.
    Uses integer LUT-based softmax, matching bob_numerics.softmax_weighted_sum
    exactly so shadow and numerics are bit-equivalent.

      q_addr: int16[HD] query vector
      k_addr_base: base of K_ALL; key t for this head at
                   k_addr_base + t*STRIDE + head*HD*2
      v_addr_base: base of V_ALL, same stride pattern
      n_keys: number of positions to attend to (1..T)
      head: which head (0..NH-1)
      out_addr: int16[HD] output
    """
    from numerics import EXP_LUT
    # Step 1: scores_raw[t] = Σ q[j] * k[t, j]  (int32)
    scores_raw = []
    for t in range(n_keys):
        k_start = k_addr_base + t * STRIDE + head * HD * 2
        dot = 0
        for j in range(HD):
            dot += load_i16(q_addr + j * 2) * load_i16(k_start + j * 2)
        scores_raw.append(dot)
    # Step 2: sf[t] = scores_raw[t] >> 14
    sf = [s >> 14 for s in scores_raw]
    max_sf = max(sf)
    # Step 3: weights via LUT
    weights = []
    for s in sf:
        delta = max_sf - s
        if delta < 0:   delta = 0
        if delta > 127: delta = 127
        weights.append(int(EXP_LUT[delta]))
    # Step 4
    w_sum = sum(weights)
    if w_sum == 0:
        w_sum = 1
    # Step 5 & 6: weighted sum and divide
    for j in range(HD):
        acc = 0
        for t in range(n_keys):
            v_start = v_addr_base + t * STRIDE + head * HD * 2
            acc += weights[t] * load_i16(v_start + j * 2)
        q = acc // w_sum
        store_i16(out_addr + j * 2, sat16(q))


def op_residual_add(h_addr, delta_addr):
    """h[i] = sat16(h[i] + delta[i]) for ED entries."""
    for i in range(ED):
        h = load_i16(h_addr + i * 2)
        d = load_i16(delta_addr + i * 2)
        store_i16(h_addr + i * 2, sat16(h + d))


def op_relu(addr, n):
    for i in range(n):
        v = load_i16(addr + i * 2)
        if v < 0:
            store_i16(addr + i * 2, 0)


def op_argmax_skip4(logits_addr):
    """Find argmax of logits[4..VS-1] (skip the 4 special tokens)."""
    best = 4
    best_v = load_i16(logits_addr + 4 * 2)
    for i in range(5, VS):
        v = load_i16(logits_addr + i * 2)
        if v > best_v:
            best_v = v
            best = i
    return best


# ─── Full forward pass ────────────────────────────────────────────
def forward_shadow(W: Weights, token_ids):
    """
    Shadow forward pass. Lays weights in MEM, runs the same sequence of
    ops a 6502 will run, returns the argmax token and logits.
    """
    addrs, shifts, total = lay_weights(W)
    assert total <= (0xBFFF - 0x4000), f"weights too big: {total}"

    T = len(token_ids)
    assert 1 <= T <= SL

    # 1. Embedding
    for t, tok in enumerate(token_ids):
        op_deshift_add_to_hidden(t, tok, addrs['te'], shifts['te'],
                                 addrs['pe'], shifts['pe'])

    # 2. Layers
    for L in range(NL):
        # 2a. Per-position pre-norm then Q/K/V
        XN = SCRATCH
        for t in range(T):
            op_rms_norm(HIDDEN + t * STRIDE, addrs[f'l{L}.n1'],
                        shifts[f'l{L}.n1'], XN)
            op_matvec(addrs[f'l{L}.q'], ED, ED, shifts[f'l{L}.q'],
                      XN, Q_ALL + t * STRIDE)
            op_matvec(addrs[f'l{L}.k'], ED, ED, shifts[f'l{L}.k'],
                      XN, K_ALL + t * STRIDE)
            op_matvec(addrs[f'l{L}.v'], ED, ED, shifts[f'l{L}.v'],
                      XN, V_ALL + t * STRIDE)

        # 2b. Causal attention at every position
        for t_q in range(T):
            for head in range(NH):
                q_addr = Q_ALL + t_q * STRIDE + head * HD * 2
                out_addr = ATT_OUT + t_q * STRIDE + head * HD * 2
                op_attn_head(q_addr, K_ALL, V_ALL, t_q + 1, head, out_addr)

        # 2c. Output projection per position + residual
        DELTA = SCRATCH
        for t in range(T):
            op_matvec(addrs[f'l{L}.proj'], ED, ED, shifts[f'l{L}.proj'],
                      ATT_OUT + t * STRIDE, DELTA)
            op_residual_add(HIDDEN + t * STRIDE, DELTA)

        # 2e. FFN per position + residual
        for t in range(T):
            YN = SCRATCH
            Z  = SCRATCH + 0x80          # ED * 2 = 64 bytes, use 128 to be safe
            W2 = SCRATCH + 0x180         # FF * 2 = 128 bytes
            op_rms_norm(HIDDEN + t * STRIDE, addrs[f'l{L}.n2'],
                        shifts[f'l{L}.n2'], YN)
            op_matvec_bias(addrs[f'l{L}.fc1_w'], addrs[f'l{L}.fc1_b'],
                           FF, ED, shifts[f'l{L}.fc1_w'], YN, Z)
            op_relu(Z, FF)
            op_matvec_bias(addrs[f'l{L}.fc2_w'], addrs[f'l{L}.fc2_b'],
                           ED, FF, shifts[f'l{L}.fc2_w'], Z, W2)
            op_residual_add(HIDDEN + t * STRIDE, W2)

    # 3. Final norm + output projection (last position only)
    Y = SCRATCH
    LOGITS = SCRATCH + 0x100   # 256 bytes for VS=128 int16
    op_rms_norm(HIDDEN + (T - 1) * STRIDE, addrs['norm'],
                shifts['norm'], Y)
    op_matvec(addrs['out'], VS, ED, shifts['out'], Y, LOGITS, post_shift=0)

    logits = np.array([load_i16(LOGITS + i * 2) for i in range(VS)],
                      dtype=np.int16)
    tok = op_argmax_skip4(LOGITS)
    return tok, logits


# ─── Self-test ─────────────────────────────────────────────────────
if __name__ == '__main__':
    from numerics import forward as ref_forward, SEP
    from test_numerics import make_synthetic_weights

    print("Shadow v3 parity test vs reference")
    print("=" * 60)
    tests = [
        [SEP, 10, 11, 12, SEP],
        [SEP, 20, 30, 40, 50, SEP],
        [SEP, 5, 6, 7, 8, 9, 10, 11, SEP],
        [SEP, 100, 50, 25, 75, SEP],
        [SEP, 80, SEP],
        [SEP, 44, 16, 72, SEP],
    ]
    total = 0
    passed = 0
    for seed in [0, 1, 5, 42, 99]:
        W = make_synthetic_weights(seed)
        for tokens in tests:
            total += 1
            tok_ref, lg_ref = ref_forward(W, tokens)
            tok_sh,  lg_sh  = forward_shadow(W, tokens)
            if tok_ref == tok_sh and np.array_equal(lg_ref, lg_sh):
                passed += 1
            else:
                diffs = int((lg_ref != lg_sh).sum())
                print(f"  FAIL seed={seed} {tokens}: ref={tok_ref} shadow={tok_sh} "
                      f"({diffs} logit diffs)")
                if diffs > 0:
                    idx = int(np.where(lg_ref != lg_sh)[0][0])
                    print(f"    first diff at logit[{idx}]: "
                          f"ref={lg_ref[idx]} shadow={lg_sh[idx]}")
    print(f"\n  {passed}/{total} cases bit-exact")
