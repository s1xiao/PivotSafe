#!/usr/bin/env python3
"""Download and prepare SSP image-detector checkpoints from the official shared archive."""
from __future__ import annotations

import argparse
import json
import shutil
import tempfile
import zipfile
from pathlib import Path

import requests


DEFAULT_URL = "https://www.dropbox.com/scl/fo/2xdvtew4rjrrsl6cseq30/AElVHGO84W1DSmlImMg1ruM?dl=1&rlkey=e1a2hnzh62wuuxrnbkfv6f90v"
SUPPORTED_VARIANTS = {
    "adm",
    "sd4",
    "sd5",
    "vqdm",
    "glide",
    "biggan",
    "wukong",
    "midjourney",
}


def _sniff_file_type(path: Path) -> str:
    with path.open("rb") as f:
        head = f.read(1024)
    if head.startswith(b"PK\x03\x04"):
        return "zip"
    lower = head.lstrip().lower()
    if lower.startswith(b"<!doctype html") or lower.startswith(b"<html") or b"<html" in lower:
        return "html"
    return "unknown"


def _archive_contains_member(archive_path: Path, member: str) -> tuple[bool, str]:
    if not archive_path.exists():
        return False, "archive file does not exist"

    sniff = _sniff_file_type(archive_path)
    if sniff == "html":
        return False, "archive looks like an HTML page (likely redirect/login/anti-bot response)"

    try:
        with zipfile.ZipFile(archive_path, "r") as zf:
            names = set(zf.namelist())
    except zipfile.BadZipFile as e:
        return False, f"bad zip file: {e}"

    if member not in names:
        top = sorted({x.split("/")[0] for x in names if x})
        return False, f"missing {member}; top entries: {top[:20]}"
    return True, "ok"


def _download_file(url: str, dst: Path, *, timeout_s: int = 60) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.NamedTemporaryFile(prefix="ssp_download_", suffix=".zip", delete=False) as tf:
        tmp_path = Path(tf.name)

    try:
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (X11; Linux x86_64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0.0.0 Safari/537.36"
            )
        }
        with requests.get(url, stream=True, timeout=timeout_s, allow_redirects=True, headers=headers) as resp:
            resp.raise_for_status()
            total = int(resp.headers.get("Content-Length") or 0)
            written = 0
            with tmp_path.open("wb") as f:
                for chunk in resp.iter_content(chunk_size=1024 * 1024):
                    if not chunk:
                        continue
                    f.write(chunk)
                    written += len(chunk)
                    if total:
                        pct = written * 100 / total
                        print(f"downloaded {written}/{total} bytes ({pct:.1f}%)", flush=True)
                    else:
                        print(f"downloaded {written} bytes", flush=True)

        file_type = _sniff_file_type(tmp_path)
        if file_type == "html":
            preview = tmp_path.read_text(encoding="utf-8", errors="ignore")[:240].replace("\n", " ")
            raise RuntimeError(
                "downloaded content is HTML rather than a zip archive. "
                "This usually means cloud-disk redirect/permission page was downloaded instead of weights. "
                f"preview={preview!r}"
            )
        if file_type != "zip":
            raise RuntimeError(f"downloaded file header is not zip: type={file_type}")

        shutil.move(str(tmp_path), str(dst))
    finally:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)


def _extract_variant_checkpoint(archive_path: Path, out_root: Path, variant: str) -> Path:
    member = f"{variant}/Net_epoch_best.pth"
    out_root.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(archive_path, "r") as zf:
        names = set(zf.namelist())
        if member not in names:
            raise FileNotFoundError(
                f"checkpoint entry not found in archive: {member}. "
                f"available top entries: {sorted({x.split('/')[0] for x in names if x})[:20]}"
            )
        zf.extract(member, path=out_root)

    ckpt = out_root / member
    if not ckpt.exists():
        raise FileNotFoundError(f"extracted checkpoint missing: {ckpt}")
    return ckpt


def _update_config(config_path: Path, checkpoint_path: Path, project_root: Path) -> None:
    cfg = {}
    if config_path.exists():
        cfg = json.loads(config_path.read_text(encoding="utf-8-sig"))

    cfg["checkpoint_path"] = str(checkpoint_path.relative_to(project_root)).replace("\\", "/")
    cfg.setdefault("device", None)
    cfg.setdefault("batch_size", 8)
    cfg.setdefault("threshold", 0.5)

    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(json.dumps(cfg, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Download SSP checkpoint and wire configs/image_detector.json")
    parser.add_argument("--url", default=DEFAULT_URL, help="SSP archive URL (default: official Dropbox share link)")
    parser.add_argument("--variant", default="adm", choices=sorted(SUPPORTED_VARIANTS), help="checkpoint variant")
    parser.add_argument(
        "--archive-path",
        default=None,
        help="local archive path; if omitted, download to checkpoints/image/ssp_pretrained_dropbox.zip",
    )
    parser.add_argument("--force", action="store_true", help="redownload archive even if it already exists")
    parser.add_argument("--no-update-config", action="store_true", help="do not update configs/image_detector.json")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    archive_path = Path(args.archive_path) if args.archive_path else (project_root / "checkpoints" / "image" / "ssp_pretrained_dropbox.zip")
    member = f"{args.variant}/Net_epoch_best.pth"

    need_download = args.force or not archive_path.exists()
    if archive_path.exists() and not args.force:
        ok, reason = _archive_contains_member(archive_path, member)
        if ok:
            print(f"archive already exists and looks valid, skip download: {archive_path}", flush=True)
        else:
            print(f"existing archive is invalid ({reason}), removing and redownloading...", flush=True)
            archive_path.unlink(missing_ok=True)
            need_download = True

    if need_download:
        print(f"downloading archive -> {archive_path}", flush=True)
        _download_file(args.url, archive_path)

    ok, reason = _archive_contains_member(archive_path, member)
    if not ok:
        raise RuntimeError(
            f"archive validation failed after download: {reason}. "
            "If you are on a network-mounted filesystem, try:\n"
            "1) python -m scripts.download_ssp_checkpoint --variant adm --force --archive-path /tmp/ssp_pretrained_dropbox.zip\n"
            "2) then rerun without --archive-path to sync into project checkpoints."
        )

    out_root = project_root / "checkpoints" / "image" / "ssp"
    ckpt = _extract_variant_checkpoint(archive_path, out_root, args.variant)
    print(f"checkpoint ready: {ckpt}", flush=True)

    if not args.no_update_config:
        config_path = project_root / "configs" / "image_detector.json"
        _update_config(config_path, ckpt, project_root)
        print(f"updated config: {config_path}", flush=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
