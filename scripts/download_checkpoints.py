"""Download the verified benchmark checkpoints and check their SHA-256 hashes."""
from pathlib import Path
import hashlib
import json
import urllib.request
R = Path(__file__).resolve().parents[1]
for item in json.loads((R / "checkpoints.json").read_text())["artifacts"]:
    target = R / item["path"]
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and hashlib.sha256(target.read_bytes()).hexdigest() == item["sha256"]:
        print("Verified existing", target.name)
        continue
    if target.exists():
        raise RuntimeError(f"{target} differs from the release; move it aside before downloading")
    temporary = target.with_suffix(".download")
    urllib.request.urlretrieve(item["url"], temporary)
    if hashlib.sha256(temporary.read_bytes()).hexdigest() != item["sha256"]:
        temporary.unlink()
        raise RuntimeError(f"Checksum mismatch for {target.name}")
    temporary.replace(target)
    print("Downloaded and verified", target.name)
