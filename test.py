#!/usr/bin/env python3
"""
Run all Soul Player C64 tests. Exit 0 if all pass.

Usage:
    python test.py           # run everything
    python test.py --quick   # skip slow 6502 tests
"""
import sys, os, argparse, time
from pathlib import Path

# add src/ to path
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

# track results
results = []


def section(name):
    print(f"\n{'─' * 55}")
    print(f"  {name}")
    print(f"{'─' * 55}")


def record(name, passed, total):
    results.append((name, passed, total))
    status = "PASS" if passed == total else "FAIL"
    print(f"  → {passed}/{total} {status}")


# ---------------------------------------------------------------------------
#  1. numerics sanity
# ---------------------------------------------------------------------------
def test_numerics():
    section("Numerics — forward pass sanity")
    from numerics import Weights, forward, pack_tensor, pack_bias, VS, ED, FF, NL, SL, SEP
    import numpy as np

    def make_synthetic_weights(seed=42):
        rng = np.random.default_rng(seed)
        def rand(shape, scale=0.3):
            return rng.normal(0, scale, size=shape).astype(np.float32)
        W = Weights()
        W.te = pack_tensor(rand((VS, ED), scale=0.9))
        W.pe = pack_tensor(rand((SL, ED), scale=0.6))
        for L in range(NL):
            layer = {
                'n1': pack_tensor(np.abs(rand(ED, scale=0.1)) + 1.0),
                'q': pack_tensor(rand((ED, ED), scale=0.3)),
                'k': pack_tensor(rand((ED, ED), scale=0.3)),
                'v': pack_tensor(rand((ED, ED), scale=0.25)),
                'proj': pack_tensor(rand((ED, ED), scale=0.2)),
                'n2': pack_tensor(np.abs(rand(ED, scale=0.1)) + 1.0),
            }
            fc1_w = rand((FF, ED), scale=0.25)
            fc1_b = rand(FF, scale=0.1)
            fc2_w = rand((ED, FF), scale=0.2)
            fc2_b = rand(ED, scale=0.05)
            layer['fc1_w'] = pack_tensor(fc1_w)
            layer['fc1_b'] = pack_bias(fc1_b, layer['fc1_w']['s'])
            layer['fc2_w'] = pack_tensor(fc2_w)
            layer['fc2_b'] = pack_bias(fc2_b, layer['fc2_w']['s'])
            W.layers[L] = layer
        W.norm_w = pack_tensor(np.abs(rand(ED, scale=0.2)) + 1.0)
        W.out_w = pack_tensor(rand((VS, ED), scale=0.8))
        return W

    W = make_synthetic_weights(42)
    test_inputs = [
        [SEP, 10, 11, 12, SEP],
        [SEP, 20, 30, 40, 50, SEP],
        [SEP, 5, 6, 7, 8, 9, 10, SEP],
        [SEP, 100, 50, 25, SEP],
        [SEP, 80, SEP],
    ]
    argmax_set = set()
    for ids in test_inputs:
        tok, logits = forward(W, ids)
        argmax_set.add(tok)

    passed = 1 if len(argmax_set) >= 2 else 0
    print(f"  {len(argmax_set)} distinct argmax outputs (need ≥2)")
    record("Numerics", passed, 1)
    # Also stash make_synthetic_weights for other tests
    return make_synthetic_weights


# ---------------------------------------------------------------------------
#  2. shadow vs reference parity
# ---------------------------------------------------------------------------
def test_shadow(make_weights):
    section("Shadow — bit-exact parity vs reference")
    from numerics import forward as ref_forward, SEP
    from shadow import forward_shadow
    import numpy as np

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
        W = make_weights(seed)
        for tokens in tests:
            total += 1
            tok_ref, lg_ref = ref_forward(W, tokens)
            tok_sh, lg_sh = forward_shadow(W, tokens)
            if tok_ref == tok_sh and np.array_equal(lg_ref, lg_sh):
                passed += 1
            else:
                print(f"    FAIL seed={seed} {tokens}")
    record("Shadow parity", passed, total)


# ---------------------------------------------------------------------------
#  3. float vs int parity
# ---------------------------------------------------------------------------
def test_float_vs_int(make_weights):
    section("Float vs int — argmax agreement")
    from numerics import (
        Weights, forward as int_forward, pack_tensor, pack_bias,
        VS, ED, NH, HD, FF, NL, SL, SEP, ACT_SHIFT,
    )
    import numpy as np
    import math

    # rebuild float weights with same seed
    rng = np.random.default_rng(42)
    def rand(shape, scale=0.3):
        return rng.normal(0, scale, size=shape).astype(np.float32)

    class FW:
        pass
    fw = FW()
    fw.te = rand((VS, ED), scale=0.9)
    fw.pe = rand((SL, ED), scale=0.6)
    fw.layers = []
    for L in range(NL):
        lay = {}
        lay['n1_w'] = np.abs(rand(ED, scale=0.1)) + 1.0
        lay['q'] = rand((ED, ED), scale=0.3)
        lay['k'] = rand((ED, ED), scale=0.3)
        lay['v'] = rand((ED, ED), scale=0.25)
        lay['proj'] = rand((ED, ED), scale=0.2)
        lay['n2_w'] = np.abs(rand(ED, scale=0.1)) + 1.0
        lay['fc1_w'] = rand((FF, ED), scale=0.25)
        lay['fc1_b'] = rand(FF, scale=0.1)
        lay['fc2_w'] = rand((ED, FF), scale=0.2)
        lay['fc2_b'] = rand(ED, scale=0.05)
        fw.layers.append(lay)
    fw.norm_w = np.abs(rand(ED, scale=0.2)) + 1.0
    fw.out_w = rand((VS, ED), scale=0.8)

    W_int = make_weights(42)

    def rmsnorm_float(x, w, eps=1e-5):
        ms = (x * x).mean()
        return x * (1.0 / math.sqrt(ms + eps)) * w
    def softmax(x):
        m = x.max()
        e = np.exp(x - m)
        return e / e.sum()

    def float_forward(fw, token_ids):
        T = len(token_ids)
        h = np.zeros((T, ED), dtype=np.float64)
        for t, tok in enumerate(token_ids):
            h[t] = fw.te[tok] + fw.pe[t]
        for L in range(NL):
            lay = fw.layers[L]
            q_all = np.zeros((T, ED)); k_all = np.zeros((T, ED)); v_all = np.zeros((T, ED))
            for t in range(T):
                xn = rmsnorm_float(h[t], lay['n1_w'])
                q_all[t] = (lay['q'] @ xn) * 0.5
                k_all[t] = (lay['k'] @ xn) * 0.5
                v_all[t] = (lay['v'] @ xn) * 0.5
            att_new = np.zeros((T, ED))
            for t_q in range(T):
                for head in range(NH):
                    sl = slice(head * HD, (head + 1) * HD)
                    q = q_all[t_q, sl]; k = k_all[:t_q+1, sl]; v = v_all[:t_q+1, sl]
                    scores = k @ q / math.sqrt(HD)
                    w = softmax(scores)
                    att_new[t_q, sl] = w @ v
            for t in range(T):
                h[t] = h[t] + (lay['proj'] @ att_new[t]) * 0.5
            for t in range(T):
                yn = rmsnorm_float(h[t], lay['n2_w'])
                z = np.maximum(0, (lay['fc1_w'] @ yn + lay['fc1_b']) * 0.5)
                w2 = (lay['fc2_w'] @ z + lay['fc2_b']) * 0.5
                h[t] = h[t] + w2
        y = rmsnorm_float(h[T-1], fw.norm_w)
        logits = fw.out_w @ y
        best = int(np.argmax(logits[4:]) + 4)
        return best, logits

    tests = [
        [SEP, 10, 11, 12, SEP],
        [SEP, 20, 30, 40, 50, SEP],
        [SEP, 5, 6, 7, 8, 9, 10, SEP],
        [SEP, 100, 50, 25, SEP],
        [SEP, 80, SEP],
        [SEP, 44, 16, 72, SEP],
    ]
    matches = 0
    for ids in tests:
        f_tok, _ = float_forward(fw, ids)
        i_tok, _ = int_forward(W_int, ids)
        if f_tok == i_tok:
            matches += 1
        else:
            print(f"    mismatch on {ids}: float={f_tok} int={i_tok}")
    record("Float vs int argmax", matches, len(tests))


# ---------------------------------------------------------------------------
#  4. 6502 assembly routine tests
# ---------------------------------------------------------------------------
def test_6502_smul16():
    section("6502 — smul16 (signed 16×16 multiply)")
    from assembler import CodeBuilder, run_subroutine
    from asm_rms_norm import build_smul16, TMP, SRC16, PROD

    code_org = 0x0900
    cb = CodeBuilder(org=code_org)
    build_smul16(cb)
    code_bytes = cb.get_bytes()
    entry = cb.labels['smul16']

    cases = [
        (100, 100), (-100, 100), (100, -100), (-100, -100),
        (200, 200), (-200, 200), (1000, 1000), (-1000, -1000),
        (12345, -6789), (0, 12345), (12345, 0),
        (1, 1), (-1, -1), (32767, 32767), (-32768, 1),
    ]
    passed = 0
    for a, b in cases:
        def setup(cpu, aa=a, bb=b):
            ai = aa & 0xFFFF; bi = bb & 0xFFFF
            cpu.mem[TMP] = ai & 0xFF; cpu.mem[TMP+1] = (ai>>8) & 0xFF
            cpu.mem[SRC16] = bi & 0xFF; cpu.mem[SRC16+1] = (bi>>8) & 0xFF
        cpu = run_subroutine(code_bytes, code_org, entry, setup, max_cycles=10_000)
        got = (cpu.mem[PROD] | (cpu.mem[PROD+1]<<8) | (cpu.mem[PROD+2]<<16) | (cpu.mem[PROD+3]<<24))
        if got >= 0x80000000: got -= 0x100000000
        if got == a * b: passed += 1
    record("smul16", passed, len(cases))


def test_6502_isqrt():
    section("6502 — isqrt32 (integer square root)")
    from assembler import CodeBuilder, run_subroutine
    from asm_rms_norm import build_isqrt32, SUMSQ, RMS, T32
    from numerics import isqrt_u32

    code_org = 0x0900
    cb = CodeBuilder(org=code_org)
    build_isqrt32(cb)
    code_bytes = cb.get_bytes()
    entry = cb.labels['isqrt32']

    cases = [32400, 37249, 65025, 10000, 1, 100, 3500, 50000]
    passed = 0
    for v in cases:
        def setup(cpu, vv=v):
            for i in range(4): cpu.mem[SUMSQ+i] = (vv>>(8*i)) & 0xFF
        cpu = run_subroutine(code_bytes, code_org, entry, setup, max_cycles=100_000)
        got = cpu.mem[RMS] | (cpu.mem[RMS+1] << 8)
        if got == isqrt_u32(v): passed += 1
    record("isqrt32", passed, len(cases))


def test_6502_matvec():
    section("6502 — matvec + matvec_bias")
    import subprocess
    r = subprocess.run([sys.executable, str(ROOT / "src" / "asm_matvec.py")],
                       capture_output=True, text=True, cwd=str(ROOT / "src"))
    output = r.stdout
    # Count lines containing "OK" or "FAIL" as test results
    test_lines = [l for l in output.split('\n') if '  OK' in l or 'FAIL' in l]
    passed = sum(1 for l in test_lines if '  OK' in l)
    total = len(test_lines)
    if "ALL PASS" in output:
        print(f"  ALL PASS ({total} cases)")
    else:
        print(output)
    record("matvec", passed, total)


def test_6502_rms_norm():
    section("6502 — rms_norm")
    import subprocess
    r = subprocess.run([sys.executable, str(ROOT / "src" / "asm_rms_norm.py")],
                       capture_output=True, text=True, cwd=str(ROOT / "src"))
    output = r.stdout
    test_lines = [l for l in output.split('\n') if '  OK' in l or 'FAIL' in l]
    passed = sum(1 for l in test_lines if '  OK' in l)
    total = len(test_lines)
    if "ALL PASS" in output:
        print(f"  ALL PASS ({total} cases)")
    else:
        print(output)
    record("rms_norm", passed, total)


def test_6502_attn_head():
    section("6502 — attn_head")
    import subprocess
    r = subprocess.run([sys.executable, str(ROOT / "src" / "asm_attn_head.py")],
                       capture_output=True, text=True, cwd=str(ROOT / "src"))
    output = r.stdout
    test_lines = [l for l in output.split('\n') if '  OK' in l or 'FAIL' in l]
    passed = sum(1 for l in test_lines if '  OK' in l)
    total = len(test_lines)
    if "ALL PASS" in output:
        print(f"  ALL PASS ({total} cases)")
    else:
        print(output)
    record("attn_head", passed, total)


def test_6502_simple():
    section("6502 — embed / argmax / relu / residual")
    import subprocess
    r = subprocess.run([sys.executable, str(ROOT / "src" / "asm_simple.py")],
                       capture_output=True, text=True, cwd=str(ROOT / "src"))
    output = r.stdout
    test_lines = [l for l in output.split('\n') if '  OK' in l or 'FAIL' in l]
    passed = sum(1 for l in test_lines if '  OK' in l)
    total = len(test_lines)
    if "ALL PASS" in output:
        print(f"  ALL PASS ({total} cases)")
    else:
        print(output)
    record("embed/argmax", passed, total)


# ---------------------------------------------------------------------------
#  5. build round-trip
# ---------------------------------------------------------------------------
def test_build():
    section("Build — assemble .prg from shipped weights")
    soul = ROOT / "models" / "soul.bin"
    tok = ROOT / "models" / "tokenizer.json"
    if not soul.exists() or not tok.exists():
        print("  SKIP: models/soul.bin or models/tokenizer.json not found")
        record("Build", 0, 1)
        return

    import subprocess, tempfile
    with tempfile.TemporaryDirectory() as tmp:
        r = subprocess.run(
            [sys.executable, str(ROOT / "src" / "build.py"),
             "--soul", str(soul), "--tokenizer", str(tok), "--output", tmp],
            capture_output=True, text=True)
        prg = Path(tmp) / "soulplayer.prg"
        d64 = Path(tmp) / "soulplayer.d64"
        ok = prg.exists() and d64.exists() and prg.stat().st_size > 20000
        if ok:
            print(f"  PRG: {prg.stat().st_size} bytes")
            print(f"  D64: {d64.stat().st_size} bytes")
        else:
            print(f"  Build failed:\n{r.stdout}\n{r.stderr}")
    record("Build", 1 if ok else 0, 1)


# ---------------------------------------------------------------------------
#  MAIN
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Run all Soul Player C64 tests")
    parser.add_argument("--quick", action="store_true",
                        help="Skip slow 6502 assembly tests")
    args = parser.parse_args()

    print("=" * 55)
    print("  SOUL PLAYER C64 — TEST SUITE")
    print("=" * 55)
    t0 = time.time()

    make_weights = test_numerics()
    test_shadow(make_weights)
    test_float_vs_int(make_weights)

    if not args.quick:
        test_6502_smul16()
        test_6502_isqrt()
        test_6502_matvec()
        test_6502_rms_norm()
        test_6502_attn_head()
        test_6502_simple()
    else:
        print("\n  (skipping 6502 tests — use without --quick for full suite)")

    test_build()

    # results
    elapsed = time.time() - t0
    print(f"\n{'═' * 55}")
    print(f"  RESULTS  ({elapsed:.1f}s)")
    print(f"{'═' * 55}")
    total_passed = 0
    total_total = 0
    all_ok = True
    for name, passed, total in results:
        status = "✓" if passed == total else "✗"
        if passed != total:
            all_ok = False
        total_passed += passed
        total_total += total
        print(f"  {status} {name:30s} {passed}/{total}")
    print(f"{'─' * 55}")
    print(f"  {'ALL PASS' if all_ok else 'FAILURES DETECTED':30s} {total_passed}/{total_total}")
    sys.exit(0 if all_ok else 1)


if __name__ == '__main__':
    main()
