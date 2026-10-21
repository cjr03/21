"""The installed strategy with weights from $ALT_WEIGHTS (for paired comparison of networks)."""
import importlib.util
import os

os.environ["VF_WEIGHTS"] = os.environ["ALT_WEIGHTS"]
_p = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "strategy", "strategy.py")
_s = importlib.util.spec_from_file_location("vf_alt_strategy", _p)
_m = importlib.util.module_from_spec(_s)
_s.loader.exec_module(_m)
rank_moves = _m.rank_moves
