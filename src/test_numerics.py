#!/usr/bin/env python3
"""Synthetic weight generator for testing."""
import numpy as np
from numerics import (
    Weights, forward, pack_tensor, pack_bias,
    VS, ED, FF, NL, SL, SEP,
)


def make_synthetic_weights(seed=42):
    """Create a Weights object from random float arrays resembling a trained model."""
    rng = np.random.default_rng(seed)
    def rand(shape, scale=0.3):
        return rng.normal(0, scale, size=shape).astype(np.float32)

    W = Weights()
    W.te    = pack_tensor(rand((VS, ED), scale=0.9))
    W.pe    = pack_tensor(rand((SL, ED), scale=0.6))

    for L in range(NL):
        layer = {
            'n1': pack_tensor(np.abs(rand(ED, scale=0.1)) + 1.0),
            'q':  pack_tensor(rand((ED, ED), scale=0.3)),
            'k':  pack_tensor(rand((ED, ED), scale=0.3)),
            'v':  pack_tensor(rand((ED, ED), scale=0.25)),
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
    W.out_w  = pack_tensor(rand((VS, ED), scale=0.8))
    return W


if __name__ == '__main__':
    W = make_synthetic_weights(42)
    test_inputs = [
        [SEP, 10, 11, 12, SEP],
        [SEP, 20, 30, 40, 50, SEP],
        [SEP, 5, 6, 7, 8, 9, 10, SEP],
    ]
    argmax_set = set()
    for ids in test_inputs:
        tok, logits = forward(W, ids)
        print(f"  {ids} -> argmax={tok}")
        argmax_set.add(tok)
    print(f"\n  {len(argmax_set)} distinct outputs — {'OK' if len(argmax_set) >= 2 else 'FAIL'}")
