from __future__ import annotations

from pathlib import Path


def _valid_yolo_line(parts: list[str]) -> bool:
    if len(parts) != 5:
        return False
    try:
        cls_id = int(float(parts[0]))
        x, y, w, h = (float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4]))
    except ValueError:
        return False
    if cls_id < 0:
        return False
    if not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0):
        return False
    if not (0.0 < w <= 1.0 and 0.0 < h <= 1.0):
        return False
    return True


def lint_yolo_labels(labels_dir: str | Path) -> dict[str, int | bool]:
    path = Path(labels_dir)
    if not path.exists():
        return {
            "ok": False,
            "files": 0,
            "lines": 0,
            "invalid_lines": 0,
            "empty_files": 0,
        }

    files = sorted(path.rglob("*.txt"))
    invalid_lines = 0
    total_lines = 0
    empty_files = 0

    for f in files:
        raw = f.read_text(encoding="utf-8", errors="ignore").strip()
        if not raw:
            empty_files += 1
            continue
        for ln in raw.splitlines():
            line = ln.strip()
            if not line:
                continue
            total_lines += 1
            if not _valid_yolo_line(line.split()):
                invalid_lines += 1

    ok = invalid_lines == 0 and len(files) > 0
    return {
        "ok": ok,
        "files": len(files),
        "lines": total_lines,
        "invalid_lines": invalid_lines,
        "empty_files": empty_files,
    }
