"""Download the FineWeb-Edu sample-10BT subset to a local cache for tests + dev runs.

Usage:
    .venv/bin/python scripts/download_fineweb_local.py            # all 14 shards (~28 GB)
    .venv/bin/python scripts/download_fineweb_local.py --shards 2 # first 2 shards (~4 GB)

The download lives at ~/.cache/nanollama/datasets/fineweb-edu-sample-10BT/sample/10BT/
and is consumed by test/test_distributed_loader.py and any dev script that calls
``find_local_fineweb()`` from this module.
"""

from __future__ import annotations

import argparse
import glob
import os
import time
from pathlib import Path

LOCAL_DIR = Path.home() / ".cache" / "nanollama" / "datasets" / "fineweb-edu-sample-10BT"
SHARD_GLOB = str(LOCAL_DIR / "sample" / "10BT" / "*.parquet")


def find_local_fineweb() -> list[str]:
	"""Return sorted local parquet shards, or [] if none cached yet."""
	return sorted(glob.glob(SHARD_GLOB))


def download(num_shards: int | None = None) -> list[str]:
	from huggingface_hub import HfApi, snapshot_download

	LOCAL_DIR.mkdir(parents=True, exist_ok=True)

	if num_shards is None:
		patterns = "sample/10BT/*"
	else:
		api = HfApi()
		all_files = api.list_repo_files("HuggingFaceFW/fineweb-edu", repo_type="dataset")
		shards = sorted(f for f in all_files if f.startswith("sample/10BT/"))
		patterns = shards[:num_shards]
		if not patterns:
			raise RuntimeError("no shards matched sample/10BT/")

	print(f"Downloading FineWeb-Edu sample-10BT -> {LOCAL_DIR}")
	print(f"Pattern: {patterns}")
	t0 = time.time()
	snapshot_download(
		repo_id="HuggingFaceFW/fineweb-edu",
		repo_type="dataset",
		allow_patterns=patterns,
		local_dir=str(LOCAL_DIR),
		max_workers=8,
	)
	dt = time.time() - t0

	files = find_local_fineweb()
	total_bytes = sum(os.path.getsize(p) for p in files)
	print(f"Done. {len(files)} files, {total_bytes/1e9:.2f} GB in {dt:.1f}s "
	      f"({total_bytes/1e9/max(dt,0.1)*8:.1f} Gbps avg)")
	return files


if __name__ == "__main__":
	ap = argparse.ArgumentParser()
	ap.add_argument("--shards", type=int, default=None,
	                help="how many leading shards to fetch (default: all 14)")
	args = ap.parse_args()
	download(args.shards)
