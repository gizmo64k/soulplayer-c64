#!/usr/bin/env python3
"""Build soulplayer.prg and soulplayer.d64 from trained weights.

Usage:
    python build.py                                    # use defaults
    python build.py --soul models/soul.bin             # custom soul
    python build.py --output disk/                     # custom output dir
"""
import sys
from pathlib import Path

# add src/ to path
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from build import main

if __name__ == '__main__':
    main()
