"""One process-rank in the multi-process distributed loader test.

Driven by env vars from test/test_distributed_loader.py::test_distributed_multiprocess_real.
Writes a JSON report to $WORKER_REPORT_PATH on success.

Required env:
    JAX_NUM_PROCESSES, JAX_PROCESS_ID, JAX_COORDINATOR_ADDRESS  (consumed by dist_init)
    XLA_FLAGS=--xla_force_host_platform_device_count=K          (must be set before jax import)
    JAX_PLATFORMS=cpu                                            (CPU-only loopback)
    DP, TP                                                       (mesh shape)
    SEQ_LEN, BATCH_PER_DP, NUM_BATCHES
    FINEWEB_GLOB                                                 (parquet files)
    WORKER_REPORT_PATH                                           (where to write JSON)
"""

from __future__ import annotations

import glob
import json
import os
import sys
import traceback


def main() -> int:
	report_path = os.environ["WORKER_REPORT_PATH"]
	report = {"ok": False, "pid": int(os.environ.get("JAX_PROCESS_ID", "0"))}

	try:
		dp = int(os.environ["DP"])
		tp = int(os.environ["TP"])
		seq_len = int(os.environ["SEQ_LEN"])
		batch_per_dp = int(os.environ["BATCH_PER_DP"])
		num_batches = int(os.environ["NUM_BATCHES"])
		fineweb_glob = os.environ["FINEWEB_GLOB"]

		import numpy as np
		import jax
		from jax.sharding import Mesh

		from ueaj.dist_init import maybe_init_distributed
		from ueaj.data.distributed_loader import (
			distributed_batch_iterator,
			compute_host_loader_slots,
			make_global_shape,
		)

		ok = maybe_init_distributed(verbose=True)
		assert ok, "expected multi-process env vars"

		report["process_index"] = jax.process_index()
		report["process_count"] = jax.process_count()
		report["local_devices"] = len(jax.local_devices())
		report["global_devices"] = jax.device_count()

		assert jax.device_count() == dp * tp, (
			f"global devs={jax.device_count()} != dp*tp={dp*tp}"
		)

		# Layout: row-major over (dp, tp) -- proc i owns row i.
		devs = np.asarray(jax.devices()).reshape(dp, tp)
		mesh = Mesh(devs, axis_names=("dp", "tp"))

		ranks, slots, total = compute_host_loader_slots(mesh, ("dp",))
		report["loader_ranks"] = ranks
		report["num_slots"] = slots
		report["total_data"] = total

		shards = sorted(glob.glob(fineweb_glob))
		assert shards, f"no parquet shards at {fineweb_glob}"

		import datasets
		hf_ds = datasets.load_dataset(
			"parquet",
			data_files=shards,
			split="train",
			streaming=True,
		)

		from transformers import AutoTokenizer
		tok = AutoTokenizer.from_pretrained("moondream/starmie-v1")
		pad_id = tok.pad_token_id if tok.pad_token_id is not None else (tok.eos_token_id or 0)

		expected_shape = make_global_shape(mesh, ("dp",), batch_per_dp, seq_len)
		report["expected_global_shape"] = list(expected_shape)

		# Per-batch records: addressable shards keyed by (dp_rank, tp_rank), with
		# the first 16 token IDs as a fingerprint we can compare cross-process.
		batch_records: list[dict] = []

		with mesh:
			gen = distributed_batch_iterator(
				hf_ds, tok, mesh, ("dp",),
				batch_size_per_data_rank=batch_per_dp,
				seq_len=seq_len,
				pad_token_id=pad_id,
			)
			# Pull exactly num_batches; an extra pull would be a one-sided
			# collective if either process broke out earlier.
			for i in range(num_batches):
				tokens, doc_ids = next(gen)
				print(f"[worker {jax.process_index()}] pulled batch {i}", flush=True)
				assert tuple(tokens.shape) == expected_shape

				per_shard = {}
				for shard in tokens.addressable_shards:
					idx = np.argwhere(devs == shard.device)
					assert idx.shape == (1, 2)
					dp_rank, tp_rank = int(idx[0, 0]), int(idx[0, 1])
					row0 = np.asarray(shard.data)[0]
					per_shard[f"{dp_rank},{tp_rank}"] = row0[:16].tolist()
				batch_records.append(per_shard)

		report["batches"] = batch_records
		report["ok"] = True

	except Exception as e:
		report["error"] = repr(e)
		report["traceback"] = traceback.format_exc()
		print(report["traceback"], file=sys.stderr)
	finally:
		with open(report_path, "w") as f:
			json.dump(report, f, indent=2)

	return 0 if report["ok"] else 1


if __name__ == "__main__":
	sys.exit(main())
