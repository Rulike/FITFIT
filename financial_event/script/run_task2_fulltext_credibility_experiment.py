#!/usr/bin/env python3
"""Run task 2 fulltext evidence credibility experiment."""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.event_credibility.fulltext import main


if __name__ == "__main__":
    main()

