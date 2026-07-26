#!/usr/bin/env python3
from __future__ import annotations

import argparse
import fnmatch
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download


@dataclass(frozen=True)
class RemoteFile:
    path: str
    size: int | None = None


def parse_scenes(value: str) -> list[str]:
    return [item.strip().strip("/") for item in value.split(",") if item.strip()]


def parse_files(value: str | None) -> list[str]:
    if not value:
        return []
    return [item.strip().strip("/") for item in value.split(",") if item.strip()]


def scene_patterns(scenes: list[str]) -> list[str]:
    return [f"MTMC_Tracking_2026/{scene}/depth_maps/*.h5" for scene in scenes]


def list_depth_files(repo_id: str, scenes: list[str]) -> list[RemoteFile]:
    api = HfApi()
    selected: list[RemoteFile] = []
    for scene in scenes:
        path_in_repo = f"MTMC_Tracking_2026/{scene}/depth_maps"
        try:
            tree = api.list_repo_tree(
                repo_id=repo_id,
                repo_type="dataset",
                path_in_repo=path_in_repo,
                recursive=False,
            )
            for item in tree:
                path = getattr(item, "path", "")
                if path.endswith(".h5"):
                    selected.append(RemoteFile(path=path, size=getattr(item, "size", None)))
        except Exception as exc:  # noqa: BLE001
            print(f"scene listing failed for {scene}: {type(exc).__name__}: {exc}", flush=True)
    if selected:
        return sorted(selected, key=lambda item: item.path)

    patterns = scene_patterns(scenes)
    print("scene listing returned no files; falling back to path-only listing", flush=True)
    files = api.list_repo_files(repo_id=repo_id, repo_type="dataset")
    selected = [RemoteFile(path=f) for f in files if any(fnmatch.fnmatch(f, pattern) for pattern in patterns)]
    return sorted(selected, key=lambda item: item.path)


def is_complete(local_dir: Path, remote_file: RemoteFile, min_bytes: int, size_slack_bytes: int) -> bool:
    path = local_dir / remote_file.path
    if not path.exists() or path.stat().st_size < min_bytes:
        return False
    if remote_file.size is None:
        return True
    return path.stat().st_size + size_slack_bytes >= remote_file.size


def download_one(repo_id: str, local_dir: Path, rel_path: str, retries: int, sleep_seconds: float) -> tuple[str, str]:
    for attempt in range(1, retries + 1):
        try:
            hf_hub_download(
                repo_id=repo_id,
                repo_type="dataset",
                filename=rel_path,
                local_dir=str(local_dir),
                local_dir_use_symlinks=False,
                resume_download=True,
            )
            return rel_path, "ok"
        except Exception as exc:  # noqa: BLE001
            if attempt >= retries:
                return rel_path, f"failed: {type(exc).__name__}: {exc}"
            print(f"retry {attempt}/{retries} {rel_path}: {type(exc).__name__}: {exc}", flush=True)
            time.sleep(sleep_seconds * attempt)
    return rel_path, "failed"


def main() -> int:
    parser = argparse.ArgumentParser(description="Robust explicit-file downloader for 2026 depth maps.")
    parser.add_argument("--repo-id", default="nvidia/PhysicalAI-SmartSpaces")
    parser.add_argument("--local-dir", type=Path, default=Path("/path/to/PhysicalAI-SmartSpaces"))
    parser.add_argument("--scenes", help="Comma list such as train/Warehouse_004,val/Warehouse_020")
    parser.add_argument("--files", help="Comma list of exact repo-relative .h5 files to download")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--retries", type=int, default=8)
    parser.add_argument("--sleep-seconds", type=float, default=30.0)
    parser.add_argument("--min-bytes", type=int, default=1024 * 1024)
    parser.add_argument("--size-slack-bytes", type=int, default=4096)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    args.local_dir.mkdir(parents=True, exist_ok=True)
    scenes = parse_scenes(args.scenes or "")
    exact_files = parse_files(args.files)
    if not scenes and not exact_files:
        raise SystemExit("No scenes or files supplied")

    print(f"repo_id={args.repo_id}", flush=True)
    print(f"local_dir={args.local_dir}", flush=True)
    print(f"scenes={','.join(scenes)}", flush=True)
    print(f"files={len(exact_files)}", flush=True)
    print(f"workers={args.workers} retries={args.retries}", flush=True)
    print(f"HF_HUB_ENABLE_HF_TRANSFER={os.environ.get('HF_HUB_ENABLE_HF_TRANSFER', '')}", flush=True)

    if exact_files:
        files = [RemoteFile(path=path) for path in exact_files]
    else:
        files = list_depth_files(args.repo_id, scenes)
    remaining = [remote_file for remote_file in files if not is_complete(args.local_dir, remote_file, args.min_bytes, args.size_slack_bytes)]
    print(f"files_total={len(files)} files_remaining={len(remaining)} files_complete={len(files) - len(remaining)}", flush=True)
    for remote_file in remaining[:20]:
        local_path = args.local_dir / remote_file.path
        local_size = local_path.stat().st_size if local_path.exists() else 0
        print(f"remaining {remote_file.path} local={local_size} remote={remote_file.size}", flush=True)
    if len(remaining) > 20:
        print(f"remaining ... {len(remaining) - 20} more", flush=True)
    if args.dry_run:
        return 0

    failures: list[str] = []
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futures = [
            pool.submit(download_one, args.repo_id, args.local_dir, remote_file.path, args.retries, args.sleep_seconds)
            for remote_file in remaining
        ]
        for idx, future in enumerate(as_completed(futures), start=1):
            rel, status = future.result()
            print(f"[{idx}/{len(futures)}] {status} {rel}", flush=True)
            if status != "ok":
                failures.append(f"{rel}\t{status}")

    if failures:
        print("Failures:", file=sys.stderr, flush=True)
        for failure in failures:
            print(failure, file=sys.stderr, flush=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
