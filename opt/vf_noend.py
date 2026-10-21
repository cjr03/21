"""The installed strategy with the exact endgame solver at $ALT_ENDGAME cards (default -1 = off),
for paired comparison."""
import importlib.util
import os

os.environ["VF_ENDGAME"] = os.environ.get("ALT_ENDGAME", "-1")
_p = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "strategy", "strategy.py")
_s = importlib.util.spec_from_file_location("vf_noend_strategy", _p)
_m = importlib.util.module_from_spec(_s)
_s.loader.exec_module(_m)
rank_moves = _m.rank_moves
