"""Check the published source snapshot using repository-relative SHA-256 hashes."""
import hashlib
import json
from pathlib import Path

root = Path(__file__).resolve().parents[2]
snapshot = json.loads((root / "docs/source_snapshot.json").read_text())
changed = []
for entry in snapshot["files"]:
    file = root / entry["repository_path"]
    if not file.is_file() or hashlib.sha256(file.read_bytes()).hexdigest() != entry["published_sha256"]:
        changed.append(entry["repository_path"])
print(json.dumps({"status": "passed" if not changed else "changed",
                  "files_checked": len(snapshot["files"]), "changed_files": changed}, indent=2))
if changed:
    raise SystemExit(1)
