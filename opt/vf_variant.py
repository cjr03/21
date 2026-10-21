"""The installed strategy with settings overridden by $VARIANT, for paired comparison.

VARIANT is a comma list of strategy settings without the VF_ prefix, e.g. "DEEP=3,MARGIN=400".
Only this wrapper reads it, so the baseline in the same bench run keeps its defaults."""
import importlib.util
import os

for item in filter(None, os.environ.get("VARIANT", "").split(",")):
    k, v = item.split("=")
    os.environ[f"VF_{k.strip()}"] = v.strip()
_p = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "strategy", "strategy.py")
_s = importlib.util.spec_from_file_location("vf_variant_strategy", _p)
_m = importlib.util.module_from_spec(_s)
_s.loader.exec_module(_m)
rank_moves = _m.rank_moves
