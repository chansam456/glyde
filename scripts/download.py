"""Download the pinned public archive, verify SHA256, and safely extract it."""
import hashlib
from pathlib import Path
import urllib.request
import zipfile

SHA = "05c8b0e6f1a2757050aced55ce4bf6ab2ac9b30f2fd8ca193056812d9c621d4d"
URL = "https://physionet-open.s3.amazonaws.com/cgmacros/1.0.0/CGMacros_dateshifted365.zip"


def extract(archive, destination):
    archive, destination = Path(archive), Path(destination).resolve()
    digest = hashlib.file_digest(archive.open("rb"), "sha256").hexdigest()
    if digest != SHA:
        raise ValueError("Archive checksum mismatch")
    destination.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as z:
        for name in z.namelist():
            if not (destination / name).resolve().is_relative_to(destination):
                raise ValueError("Unsafe archive member")
        z.extractall(destination)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--archive", type=Path, default=Path("data/CGMacros.zip"))
    args = parser.parse_args()
    if not args.archive.exists():
        args.archive.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(URL, args.archive)
    extract(args.archive, "data/raw")
