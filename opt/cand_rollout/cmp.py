import sys, numpy as np
a = np.load(sys.argv[1]); b = np.load(sys.argv[2])
sa = dict(zip(a['seeds'], a['scores'])); sb = dict(zip(b['seeds'], b['scores']))
common = sorted(set(sa) & set(sb))
d = np.array([sa[s] - sb[s] for s in common])
print(f"n={len(d)} A {np.mean([sa[s] for s in common]):.1f} B {np.mean([sb[s] for s in common]):.1f} diff {d.mean():+.1f} se {d.std()/np.sqrt(len(d)):.1f}")
