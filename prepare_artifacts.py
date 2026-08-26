"""Reconstruct and verify the supplied trained checkpoint."""

import hashlib
import json
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parent
MANIFEST = ROOT / "artifacts.json"


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def reconstruct(entry):
    target = ROOT / entry["path"]
    expected_hash = entry["sha256"]
    expected_size = entry["size"]

    if target.exists():
        if target.stat().st_size == expected_size and sha256(target) == expected_hash:
            print(f"verified {entry['path']}")
            return
        raise RuntimeError(f"Existing artifact failed verification: {target}")

    temporary = target.with_name(f".{target.name}.tmp")
    temporary.parent.mkdir(parents=True, exist_ok=True)
    try:
        with temporary.open("wb") as output:
            for relative_part in entry["parts"]:
                part = ROOT / relative_part
                if not part.is_file():
                    raise FileNotFoundError(f"Missing artifact part: {part}")
                with part.open("rb") as source:
                    for block in iter(lambda: source.read(1024 * 1024), b""):
                        output.write(block)

        if temporary.stat().st_size != expected_size:
            raise RuntimeError(f"Unexpected artifact size: {entry['path']}")
        if sha256(temporary) != expected_hash:
            raise RuntimeError(f"Checksum mismatch: {entry['path']}")
        os.replace(temporary, target)
    finally:
        temporary.unlink(missing_ok=True)

    print(f"reconstructed {entry['path']}")


def main():
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    for entry in manifest["artifacts"]:
        reconstruct(entry)


if __name__ == "__main__":
    main()
