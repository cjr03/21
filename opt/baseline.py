"""Baseline strategy: the current one-card-lookahead heuristic in twentyone.py."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from twentyone import _heuristic_rank_moves as rank_moves  # noqa: E402,F401
