"""Modal smoke test for the distributed JAX data loader.

Two entrypoints:

    modal run scripts/modal_test_loader.py::prepare_cache
        Pre-downloads the FineWeb-Edu sample-10BT parquet shards and the
        moondream/starmie-v1 tokenizer into a persistent Modal Volume
        mounted at ``/hf-cache``. Run once; reused by every subsequent test.

    modal run scripts/modal_test_loader.py
        Runs the loader smoke test. If the local cache at /hf-cache contains
        the 10BT shards it streams from disk; otherwise it logs a warning
        and falls back to streaming from the HuggingFace Hub.

Env knobs (via ``modal run`` time):

    GPUS=2 DP=2 TP=1 NUM_NODES=1 NUM_BATCHES=5 BATCH_PER_DP=1 SEQ_LEN=512

Multi-host extension: bump ``NUM_NODES`` to 2 and Modal will spin up a small
i6pn cluster; each container runs ``run_loader_test`` with its own rank and
calls ``jax.distributed.initialize``.
"""

from __future__ import annotations

import os

import modal


# --------------------------------------------------------------------------
# Image + persistent HF cache volume
# --------------------------------------------------------------------------

REQUIREMENTS = [
	"jax[cuda12]==0.8.1",
	"flax==0.13.1",
	"datasets==4.4.1",
	"transformers==4.57.1",
	"huggingface_hub>=0.36.0",
	"numpy",
]

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

HF_CACHE_DIR = "/hf-cache"
LOCAL_FINEWEB_DIR = f"{HF_CACHE_DIR}/fineweb-edu-sample-10BT"
TOKENIZER_NAME = "moondream/starmie-v1"

image = (
	modal.Image.debian_slim(python_version="3.11")
	.pip_install(*REQUIREMENTS)
	.add_local_dir(REPO_ROOT, "/root/nanollama", copy=True)
	.workdir("/root/nanollama")
	.env({
		"PYTHONPATH": "/root/nanollama",
		"TRITON_ALLOW_NON_CONSTEXPR_GLOBALS": "1",
		"HF_HOME": HF_CACHE_DIR,
		"HF_DATASETS_CACHE": f"{HF_CACHE_DIR}/datasets",
	})
)

hf_cache = modal.Volume.from_name("nanollama-hf-cache", create_if_missing=True)

app = modal.App("nanollama-dist-loader-test", image=image)


# --------------------------------------------------------------------------
# Tunables (overridden via env at modal-run time)
# --------------------------------------------------------------------------

GPUS = int(os.environ.get("GPUS", "2"))
DP = int(os.environ.get("DP", "2"))
TP = int(os.environ.get("TP", "1"))
NUM_NODES = int(os.environ.get("NUM_NODES", "1"))
NUM_BATCHES = int(os.environ.get("NUM_BATCHES", "5"))
BATCH_PER_DP = int(os.environ.get("BATCH_PER_DP", "1"))
SEQ_LEN = int(os.environ.get("SEQ_LEN", "512"))


# --------------------------------------------------------------------------
# Pre-fetch entrypoint: download FineWeb-Edu 10BT shards + tokenizer
# --------------------------------------------------------------------------

@app.function(
	volumes={HF_CACHE_DIR: hf_cache},
	timeout=60 * 60 * 4,  # 10BT is ~28 GB compressed parquet
	cpu=4.0,
)
def prepare_cache() -> dict:
	"""Pre-download the 10BT subset + tokenizer into the persistent volume.

	Idempotent — re-running only fetches files that aren't already there.
	"""
	import glob
	import time

	from huggingface_hub import snapshot_download
	import transformers

	os.makedirs(LOCAL_FINEWEB_DIR, exist_ok=True)

	print(f"[prepare_cache] downloading FineWeb-Edu sample-10BT -> {LOCAL_FINEWEB_DIR}")
	t0 = time.time()
	snapshot_download(
		repo_id="HuggingFaceFW/fineweb-edu",
		repo_type="dataset",
		allow_patterns="sample/10BT/*",
		local_dir=LOCAL_FINEWEB_DIR,
	)
	dataset_dt = time.time() - t0
	parquet_files = sorted(glob.glob(f"{LOCAL_FINEWEB_DIR}/sample/10BT/*.parquet"))
	total_bytes = sum(os.path.getsize(p) for p in parquet_files)
	print(f"[prepare_cache] {len(parquet_files)} parquet files, {total_bytes/1e9:.2f} GB, {dataset_dt:.1f}s")

	print(f"[prepare_cache] caching tokenizer {TOKENIZER_NAME}")
	t0 = time.time()
	tok = transformers.AutoTokenizer.from_pretrained(TOKENIZER_NAME)
	tok_dt = time.time() - t0
	print(f"[prepare_cache] tokenizer ready in {tok_dt:.1f}s, vocab_size={tok.vocab_size}")

	hf_cache.commit()

	return {
		"parquet_files": len(parquet_files),
		"total_bytes": total_bytes,
		"dataset_seconds": dataset_dt,
		"tokenizer_seconds": tok_dt,
		"tokenizer_vocab_size": tok.vocab_size,
	}


# --------------------------------------------------------------------------
# Loader helper: prefer local cache, fall back to Hub
# --------------------------------------------------------------------------

def _open_fineweb_stream():
	"""Return (dataset, source_label). Prefers local /hf-cache mirror."""
	import glob
	import datasets

	local_glob = f"{LOCAL_FINEWEB_DIR}/sample/10BT/*.parquet"
	local_files = sorted(glob.glob(local_glob))
	if local_files:
		print(f"[loader] streaming {len(local_files)} parquet files from {LOCAL_FINEWEB_DIR}")
		ds = datasets.load_dataset(
			"parquet",
			data_files=local_files,
			split="train",
			streaming=True,
		)
		return ds, "local"

	print(
		f"[loader] WARNING: no local 10BT cache at {local_glob}; "
		"streaming from huggingface.co. Run `modal run scripts/modal_test_loader.py::prepare_cache` "
		"first to avoid network egress on every test."
	)
	ds = datasets.load_dataset(
		"HuggingFaceFW/fineweb-edu",
		name="sample-10BT",
		split="train",
		streaming=True,
	)
	return ds, "hub"


# --------------------------------------------------------------------------
# Worker
# --------------------------------------------------------------------------

@app.function(
	gpu=f"L4:{GPUS}",
	timeout=1800,
	volumes={HF_CACHE_DIR: hf_cache},
)
def run_loader_test(
	process_id: int,
	num_processes: int,
	coordinator_address: str,
	dp: int,
	tp: int,
	gpus: int,
	num_batches: int,
	batch_per_dp: int,
	seq_len: int,
) -> dict:
	import socket
	import time

	os.environ["JAX_COORDINATOR_ADDRESS"] = coordinator_address
	os.environ["JAX_NUM_PROCESSES"] = str(num_processes)
	os.environ["JAX_PROCESS_ID"] = str(process_id)

	import jax
	import numpy as np
	import transformers
	from jax.sharding import Mesh

	from ueaj.dist_init import maybe_init_distributed
	from ueaj.data.distributed_loader import (
		distributed_batch_iterator,
		compute_host_loader_slots,
		make_global_shape,
	)

	maybe_init_distributed(verbose=True)

	print(
		f"[pid={jax.process_index()}/{jax.process_count()}] "
		f"host={socket.gethostname()} local_devs={len(jax.local_devices())} "
		f"global_devs={jax.device_count()}"
	)

	assert jax.device_count() == dp * tp, (
		f"expected {dp*tp} global devices, got {jax.device_count()}"
	)

	devices = np.asarray(jax.devices()).reshape(dp, tp)
	mesh = Mesh(devices, axis_names=("data", "tp"))

	loader_ranks, num_slots, total_data = compute_host_loader_slots(mesh, ("data",))
	print(
		f"[pid={jax.process_index()}] data_ranks_owned={loader_ranks} "
		f"loader_slots={num_slots} total_data={total_data}"
	)

	tok = transformers.AutoTokenizer.from_pretrained(TOKENIZER_NAME)
	tok.model_max_length = seq_len
	pad_id = tok.pad_token_id if tok.pad_token_id is not None else (tok.eos_token_id or 0)

	ds, source = _open_fineweb_stream()

	with mesh:
		it = distributed_batch_iterator(
			ds, tok, mesh, ("data",),
			batch_size_per_data_rank=batch_per_dp,
			seq_len=seq_len,
			pad_token_id=pad_id,
		)

		global_batch, _ = make_global_shape(mesh, ("data",), batch_per_dp, seq_len)
		print(f"[pid={jax.process_index()}] expected global_batch={global_batch}")

		seen = []
		t0 = time.time()
		for i, (tokens, docs) in enumerate(it):
			if i >= num_batches:
				break
			shards_by_dp = {}
			for shard in tokens.addressable_shards:
				dev = shard.device
				idx = np.argwhere(devices == dev)
				assert idx.shape == (1, 2)
				dp_rank, tp_rank = int(idx[0, 0]), int(idx[0, 1])
				row = np.asarray(shard.data)
				shards_by_dp.setdefault(dp_rank, {})[tp_rank] = row

			for dp_rank, by_tp in shards_by_dp.items():
				ref = next(iter(by_tp.values()))
				for tp_rank, row in by_tp.items():
					if not np.array_equal(row, ref):
						return {
							"ok": False,
							"reason": f"tp-replication mismatch at dp={dp_rank} tp={tp_rank}",
						}

			dp_ranks_seen = sorted(shards_by_dp.keys())
			rows = [shards_by_dp[r][next(iter(shards_by_dp[r]))] for r in dp_ranks_seen]
			seen.append((dp_ranks_seen, rows, int(tokens.shape[0]) * int(tokens.shape[1])))

			print(
				f"[pid={jax.process_index()}] batch {i} "
				f"global_shape={tokens.shape} "
				f"dp_ranks_local={dp_ranks_seen} "
				f"first_token={[int(r[0,0]) for r in rows]}"
			)

		dt = time.time() - t0

	any_diff = False
	for dp_ranks, rows, _ in seen:
		if len(rows) >= 2 and not np.array_equal(rows[0], rows[1]):
			any_diff = True
			break

	report = {
		"ok": True,
		"pid": jax.process_index(),
		"data_source": source,
		"loader_ranks": loader_ranks,
		"num_slots": num_slots,
		"total_data": total_data,
		"batches": len(seen),
		"local_dp_diff_observed": any_diff,
		"elapsed_s": dt,
		"tokens_consumed_global": sum(t for _, _, t in seen),
	}
	print(f"[pid={jax.process_index()}] DONE {report}")
	return report


# --------------------------------------------------------------------------
# Local entrypoint: spawn N processes (one per "host") and join.
# --------------------------------------------------------------------------

@app.local_entrypoint()
def main():
	"""Fan out NUM_NODES processes; each runs ``run_loader_test``."""
	num_processes = NUM_NODES
	if num_processes == 1:
		coord = "127.0.0.1:0"
		results = [
			run_loader_test.remote(
				process_id=0,
				num_processes=1,
				coordinator_address=coord,
				dp=DP,
				tp=TP,
				gpus=GPUS,
				num_batches=NUM_BATCHES,
				batch_per_dp=BATCH_PER_DP,
				seq_len=SEQ_LEN,
			)
		]
	else:
		raise NotImplementedError(
			"Multi-node smoke test requires Modal Cluster / i6pn wiring. "
			"Stay with NUM_NODES=1 for the first pass."
		)

	all_ok = all(r.get("ok") for r in results)
	print("\n========================= TEST SUMMARY =========================")
	for r in results:
		print(r)
	print("================================================================")
	if all_ok:
		print("PASS")
	else:
		print("FAIL")
		raise SystemExit(1)
