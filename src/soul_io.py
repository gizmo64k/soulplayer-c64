#!/usr/bin/env python3
"""Soul file format v3 with per-tensor shifts and prescaled int32 biases."""
import struct
import numpy as np
from numerics import Weights, VS, ED, FF, NL, SL


def tensor_spec():
    spec = [('te', 'w', (VS, ED)), ('pe', 'w', (SL, ED))]
    for L in range(NL):
        spec += [
            (f'l{L}.n1',    'w', (ED,)),
            (f'l{L}.q',     'w', (ED, ED)),
            (f'l{L}.k',     'w', (ED, ED)),
            (f'l{L}.v',     'w', (ED, ED)),
            (f'l{L}.proj',  'w', (ED, ED)),
            (f'l{L}.n2',    'w', (ED,)),
            (f'l{L}.fc1_w', 'w', (FF, ED)),
            (f'l{L}.fc1_b', 'b', (FF,)),
            (f'l{L}.fc2_w', 'w', (ED, FF)),
            (f'l{L}.fc2_b', 'b', (ED,)),
        ]
    spec += [('norm', 'w', (ED,)), ('out', 'w', (VS, ED))]
    return spec


def write_soul_v3(path, tensors):
    buf = bytearray()
    for name, kind, shape in tensor_spec():
        e = tensors[name]
        rows = shape[0]
        cols = shape[1] if len(shape) == 2 else 1
        if kind == 'w':
            buf += struct.pack('<BHHb', 0, rows, cols, e['s'])
            buf += e['q'].tobytes()
        else:
            buf += struct.pack('<BHHb', 1, rows, cols, e['s'])
            buf += e['q16'].astype('<i2').tobytes()
    open(path, 'wb').write(bytes(buf))
    return len(buf)


def read_soul_v3(path) -> Weights:
    data = open(path, 'rb').read()
    off = 0
    parsed = {}
    for name, kind, shape in tensor_spec():
        k, rows, cols, s = struct.unpack_from('<BHHb', data, off)
        off += 6
        if k == 0:
            n = rows * cols
            arr = np.frombuffer(data[off:off + n], dtype=np.int8).copy()
            off += n
            if len(shape) == 2: arr = arr.reshape(shape)
            parsed[name] = {'q': arr, 's': s}
        else:
            n = rows * cols
            arr = np.frombuffer(data[off:off + n * 2], dtype='<i2').copy()
            off += n * 2
            parsed[name] = {'q16': arr, 's': s}

    W = Weights()
    W.te = parsed['te']
    W.pe = parsed['pe']
    W.norm_w = parsed['norm']
    W.out_w = parsed['out']
    for L in range(NL):
        W.layers[L] = {
            'n1': parsed[f'l{L}.n1'], 'q': parsed[f'l{L}.q'],
            'k':  parsed[f'l{L}.k'],  'v': parsed[f'l{L}.v'],
            'proj': parsed[f'l{L}.proj'], 'n2': parsed[f'l{L}.n2'],
            'fc1_w': parsed[f'l{L}.fc1_w'], 'fc1_b': parsed[f'l{L}.fc1_b'],
            'fc2_w': parsed[f'l{L}.fc2_w'], 'fc2_b': parsed[f'l{L}.fc2_b'],
        }
    return W


def soul_size(tensors):
    n = 0
    for name, kind, shape in tensor_spec():
        n += 6  # header
        sz = shape[0] * (shape[1] if len(shape) == 2 else 1)
        n += sz * (1 if kind == 'w' else 4)
    return n


if __name__ == '__main__':
    from test_numerics import make_synthetic_weights
    from numerics import forward, SEP

    W1 = make_synthetic_weights(seed=1)
    tensors = {'te': W1.te, 'pe': W1.pe, 'norm': W1.norm_w, 'out': W1.out_w}
    for L in range(NL):
        lay = W1.layers[L]
        for k, v in lay.items():
            tensors[f'l{L}.{k}'] = v

    sz = write_soul_v3('/tmp/v3.soul', tensors)
    print(f"wrote {sz} bytes")

    W2 = read_soul_v3('/tmp/v3.soul')
    for ids in [[SEP, 10, 11, 12, SEP], [SEP, 5, 6, 7, 8, 9, SEP]]:
        t1, l1 = forward(W1, ids)
        t2, l2 = forward(W2, ids)
        ok = t1 == t2 and np.array_equal(l1, l2)
        print(f"  {ids}: t1={t1} t2={t2} {'OK' if ok else 'FAIL'}")
