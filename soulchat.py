#!/usr/bin/env python3
"""
SOUL CHAT — talk to your soul directly in Python.

No emulator needed. This runs the exact same integer inference
the C64 does, bit for bit, just a few million times faster.

Usage:
    python soulchat.py                          # use default soul
    python soulchat.py --soul models/soul.bin   # use custom soul

Type a message, press enter, wait. Type 'q' to quit.
"""
import sys, json, argparse
from pathlib import Path

# add src/ to path
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from numerics import forward, VS, ED, SL, SEP, END, PAD
from soul_io import read_soul_v3


def load_tokenizer(path):
    tok = json.load(open(path))
    vocab = tok['vocab']
    merges = tok['merges']
    id_to_str = {v: k for k, v in vocab.items()}
    return vocab, merges, id_to_str


def encode(text, vocab, merges):
    """Encode text to token IDs — character-level, matching the C64 encoder.
    Each character maps directly to its vocab token, then BPE merges
    are applied on the token ID array."""
    ids = []
    for ch in text.lower():
        if ch in vocab:
            ids.append(vocab[ch])
        # else: skip (unknown char)

    # apply BPE merges on token IDs
    for a, b in merges:
        merged = a + b
        if merged not in vocab:
            continue
        a_id = vocab.get(a)
        b_id = vocab.get(b)
        m_id = vocab[merged]
        if a_id is None or b_id is None:
            continue
        new_ids = []; i = 0
        while i < len(ids):
            if i < len(ids) - 1 and ids[i] == a_id and ids[i + 1] == b_id:
                new_ids.append(m_id); i += 2
            else:
                new_ids.append(ids[i]); i += 1
        ids = new_ids
    return ids


def decode(ids, id_to_str):
    """Decode token IDs back to text."""
    parts = []
    for i in ids:
        s = id_to_str.get(i, '')
        if s in ('<PAD>', '<SEP>', '<UNK>', '<END>'):
            continue
        parts.append(s)
    return ''.join(parts)


def generate(W, token_ids, id_to_str, max_tokens=20, stream=True):
    """Autoregressive generation — same loop as the C64's run_inference."""
    ids = list(token_ids)
    generated = []

    for _ in range(max_tokens):
        if len(ids) >= SL:
            break

        tok_id, logits = forward(W, ids)

        if tok_id in (PAD, SEP, END):
            break

        ids.append(tok_id)
        generated.append(tok_id)

        if stream:
            token_str = id_to_str.get(tok_id, '?')
            if token_str not in ('<PAD>', '<SEP>', '<UNK>', '<END>'):
                print(token_str, end='', flush=True)

    if stream:
        print()

    return generated


def main():
    parser = argparse.ArgumentParser(description="Chat with a soul")
    parser.add_argument("--soul", default=str(ROOT / "models" / "soul.bin"),
                        help="Path to soul .bin file")
    parser.add_argument("--tokenizer", default=str(ROOT / "models" / "tokenizer.json"),
                        help="Path to tokenizer .json file")
    args = parser.parse_args()

    if not Path(args.soul).exists():
        print(f"  Soul not found: {args.soul}")
        print(f"  Train one first: python train.py")
        sys.exit(1)
    if not Path(args.tokenizer).exists():
        print(f"  Tokenizer not found: {args.tokenizer}")
        sys.exit(1)

    print()
    print("   .---------. ")
    print("  |  O     O  |")
    print("  |     V     |")
    print("  |..|-----|..|")
    print()
    print("  SOUL CHAT")
    print(f"  {Path(args.soul).name} loaded")
    print()
    print("  type a message. lowercase only.")
    print("  type 'q' to quit.")
    print()

    W = read_soul_v3(args.soul)
    vocab, merges, id_to_str = load_tokenizer(args.tokenizer)

    while True:
        try:
            user_input = input("YOU> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not user_input:
            continue
        if user_input.lower() == 'q':
            break

        input_ids = encode(user_input, vocab, merges)
        prompt = [SEP] + input_ids + [SEP]

        print("C64> ", end='', flush=True)
        generate(W, prompt, id_to_str, max_tokens=20, stream=True)

    print()
    print("  -- the only winning move is love!")
    print()


if __name__ == '__main__':
    main()
