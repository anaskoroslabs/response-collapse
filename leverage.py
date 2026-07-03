import sys, pathlib; sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))
import json, hashlib
sums=json.load(open("SHA256SUMS.json"))
bad=[p for p,h in sums.items() if pathlib.Path(p).exists() and
     hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()!=h]
assert not bad, f"hash mismatch: {bad}"
print(f"manifest hashes OK ({len(sums)} files)")
