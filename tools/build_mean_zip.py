"""Package the empirical mean predictor without a data payload or API credentials."""

from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo


def build_zip(output: Path | None = None) -> Path:
    root = Path(__file__).resolve().parents[1]
    output = output or root / "dist/empirical_mean.zip"
    output.parent.mkdir(parents=True, exist_ok=True)
    # An explicit allowlist keeps local files and credentials out of the ZIP.
    files = {name: root / "empirical_mean" / name for name in ("model.py", "README.md")}
    files["LICENSE"] = root / "LICENSE"
    with ZipFile(output, "w", compression=ZIP_DEFLATED) as archive:
        for name, source in files.items():
            entry = ZipInfo(name)
            entry.compress_type = ZIP_DEFLATED
            entry.external_attr = 0o100644 << 16
            archive.writestr(entry, source.read_bytes())
    return output


if __name__ == "__main__":
    path = build_zip()
    print(f"Wrote {path} ({path.stat().st_size:,} bytes)")
