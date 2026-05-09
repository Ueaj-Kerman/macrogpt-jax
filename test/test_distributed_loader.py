"""Tests for the distributed data loader against the user's spec.

Spec (verbatim from user):

    "we only want to load a number of samples proportional to the number of
     first ranks along all the non-data axis for the devices the host has
     access to. For example, if a single host has 2 / 8 GPUs and we're doing
     2x DP and 4x TP, and the single host has both GPUs of TP rank 0, then it
     would load 2 (packed) batches, and all the others would load nothing
     (or empty preset tensors that are then synced w/ the main one async)"

Layer 1 (pure logic, no JAX cluster) targets ``_slots_from_coords`` directly.
Layer 2 (synthetic-data integration) is gated on JAX exposing >=8 devices and
needs ``XLA_FLAGS=--xla_force_host_platform_device_count=8 JAX_PLATFORMS=cpu``
to be set in the environment before pytest imports JAX.

Layer 3 (true multi-process loopback) lives in
``scripts/test_dist_loader_multiprocess.py`` and is invoked manually.
"""

from __future__ import annotations

import itertools
import os

import numpy as np
import pytest

from ueaj.data.distributed_loader import _slots_from_coords


# ---------------------------------------------------------------------------
# Layer 1: pure slot-counting tests. No JAX cluster needed.
# ---------------------------------------------------------------------------


def test_single_device_no_data_axis():
	# Degenerate: 1 axis, 1 device, data_axes=().
	# Single device at coord (0,) is a slot at rank 0 (everything-zero non-data path).
	r, slots, total = _slots_from_coords(["only"], [1], [], [(0,)])
	assert (r, slots, total) == ([0], 1, 1)


def test_single_device_data_axis():
	# 1D mesh with size 1, named the data axis. Device at (0,) is a loader slot.
	r, slots, total = _slots_from_coords(["data"], [1], ["data"], [(0,)])
	assert (r, slots, total) == ([0], 1, 1)


def test_dp_only_every_device_is_a_loader():
	# 1D DP=4 on a single host: every device is a loader, ranks 0..3.
	coords = [(0,), (1,), (2,), (3,)]
	r, slots, total = _slots_from_coords(["data"], [4], ["data"], coords)
	assert r == [0, 1, 2, 3]
	assert slots == 4
	assert total == 4


def test_tp_only_only_rank0_is_a_loader():
	# 1D TP=4 with NO data axis: only the device at coord (0,) is a slot.
	# data world size = 1, all loader slots write to rank 0.
	coords = [(0,), (1,), (2,), (3,)]
	r, slots, total = _slots_from_coords(["tp"], [4], [], coords)
	assert r == [0]
	assert slots == 1
	assert total == 1


def test_users_example_dp2_tp4_host_with_two_tp0_devices():
	# User's exact example:
	#   8 GPUs, 2x DP × 4x TP, host owns 2/8 GPUs -- both at TP=0.
	#   Expected: this host loads 2 packed batches at data-ranks {0, 1}.
	coords = [(0, 0), (1, 0)]  # (dp=0, tp=0) and (dp=1, tp=0)
	r, slots, total = _slots_from_coords(
		["dp", "tp"], [2, 4], ["dp"], coords
	)
	assert r == [0, 1]
	assert slots == 2
	assert total == 2


def test_users_example_dp2_tp4_host_with_no_tp0_devices():
	# Same mesh, but the host owns only TP>0 devices.
	# Expected: this host loads NOTHING (empty placeholders downstream).
	coords = [(0, 1), (0, 2)]
	r, slots, total = _slots_from_coords(
		["dp", "tp"], [2, 4], ["dp"], coords
	)
	assert r == []
	assert slots == 0
	assert total == 2


def test_dp2_tp4_host_with_one_tp0_one_tp_other():
	# Mixed: host owns one TP=0 device and one TP>0 device.
	# Only the TP=0 one is a loader slot.
	coords = [(0, 0), (0, 3)]
	r, slots, total = _slots_from_coords(
		["dp", "tp"], [2, 4], ["dp"], coords
	)
	assert r == [0]
	assert slots == 1
	assert total == 2


def test_3d_mesh_dp_fsdp_tp_two_data_axes():
	# (dp=2, fsdp=2, tp=2) on a single host owning all 8 devices,
	# data_axes=("dp", "fsdp"). Slots = devices where tp=0 -> 4 slots.
	# Ranks linearise as dp * fsdp_size + fsdp:
	#   (0,0,0)->0, (0,1,0)->1, (1,0,0)->2, (1,1,0)->3
	coords = list(itertools.product([0, 1], [0, 1], [0, 1]))
	r, slots, total = _slots_from_coords(
		["dp", "fsdp", "tp"], [2, 2, 2], ["dp", "fsdp"], coords
	)
	assert r == [0, 1, 2, 3]
	assert slots == 4
	assert total == 4


def test_data_axes_order_changes_rank_linearisation():
	# Same physical coords, different data_axes ordering -> different ranks.
	coords = [(0, 1, 0), (1, 0, 0)]  # both at tp=0
	r_dp_first, _, _ = _slots_from_coords(
		["dp", "fsdp", "tp"], [2, 2, 2], ["dp", "fsdp"], coords
	)
	r_fsdp_first, _, _ = _slots_from_coords(
		["dp", "fsdp", "tp"], [2, 2, 2], ["fsdp", "dp"], coords
	)
	# dp-first: (0,1)->0*2+1=1; (1,0)->1*2+0=2
	assert r_dp_first == [1, 2]
	# fsdp-first: (1,0)->1*2+0=2; (0,1)->0*2+1=1
	assert r_fsdp_first == [1, 2]
	# Set is the same here (sorted), but the *meaning* is different:
	# rank 1 in dp-first = (dp=0,fsdp=1); rank 1 in fsdp-first = (fsdp=0,dp=1).
	# We can't observe the difference from just the rank numbers, but we can
	# check via a coord that doesn't symmetrise:
	asym = [(0, 1, 0)]
	r1, _, _ = _slots_from_coords(["dp","fsdp","tp"], [2,3,2], ["dp","fsdp"], asym)
	r2, _, _ = _slots_from_coords(["dp","fsdp","tp"], [2,3,2], ["fsdp","dp"], asym)
	assert r1 == [1]   # 0*3 + 1 = 1
	assert r2 == [2]   # 1*2 + 0 = 2


def test_total_data_size_is_product_of_data_axes():
	# total = prod(mesh.shape[a] for a in data_axes), independent of host coords.
	_, _, t1 = _slots_from_coords(["a", "b", "c"], [3, 5, 7], ["a", "b"], [])
	_, _, t2 = _slots_from_coords(["a", "b", "c"], [3, 5, 7], ["b"], [])
	_, _, t3 = _slots_from_coords(["a", "b", "c"], [3, 5, 7], [], [])
	assert t1 == 15
	assert t2 == 5
	assert t3 == 1


def test_no_devices_owned_returns_empty():
	# A host that owns zero devices in this mesh.
	r, slots, total = _slots_from_coords(["dp", "tp"], [2, 4], ["dp"], [])
	assert r == []
	assert slots == 0
	assert total == 2


def test_duplicate_data_ranks_dedupe_in_unique_list_but_count_devices():
	# In a real cluster you'd never have two devices at the same coord, but the
	# helper deduplicates anyway. ``slots`` still counts the raw devices.
	coords = [(0, 0), (0, 0), (1, 0)]
	r, slots, total = _slots_from_coords(
		["dp", "tp"], [2, 4], ["dp"], coords
	)
	assert r == [0, 1]
	assert slots == 3
	assert total == 2


@pytest.mark.parametrize(
	"dp,tp,host_tp_ranks,expected_slot_count,expected_ranks",
	[
		(2, 4, [0],       2, [0, 1]),     # both dp values, only at tp=0
		(2, 4, [1],       0, []),         # both dp values at tp=1 -> no loaders
		(4, 2, [0],       4, [0,1,2,3]),  # 4-way DP, all dp at tp=0
		(8, 1, [0],       8, list(range(8))),  # pure DP
		(1, 8, [0],       1, [0]),        # pure TP, only rank 0
	],
)
def test_parametrized_dp_tp_layouts(
	dp, tp, host_tp_ranks, expected_slot_count, expected_ranks
):
	"""For each tp value the host owns, it owns one device per dp coord."""
	coords = [(d, t) for d in range(dp) for t in host_tp_ranks]
	r, slots, total = _slots_from_coords(
		["dp", "tp"], [dp, tp], ["dp"], coords
	)
	assert r == expected_ranks
	assert slots == expected_slot_count
	assert total == dp


# ---------------------------------------------------------------------------
# Layer 2: integration on fake CPU devices. Skipped unless JAX exposes >= 8.
# ---------------------------------------------------------------------------

import jax  # noqa: E402  (import-after-tests is intentional; envs set before tests)


_HAS_8_DEVICES = len(jax.devices()) >= 8


import glob  # noqa: E402
from pathlib import Path  # noqa: E402

_LOCAL_FINEWEB_GLOB = str(
	Path.home() / ".cache" / "nanollama" / "datasets"
	/ "fineweb-edu-sample-10BT" / "sample" / "10BT" / "*.parquet"
)
_LOCAL_SHARDS = sorted(glob.glob(_LOCAL_FINEWEB_GLOB))


@pytest.mark.skipif(
	not _HAS_8_DEVICES,
	reason=(
		"Layer 2 needs >=8 devices. Run with "
		"XLA_FLAGS='--xla_force_host_platform_device_count=8' JAX_PLATFORMS=cpu "
		"pytest test/test_distributed_loader.py -k integration"
	),
)
@pytest.mark.skipif(
	not _LOCAL_SHARDS,
	reason=(
		"Layer 2 streams from a local FineWeb-Edu mirror. "
		"Run `.venv/bin/python scripts/download_fineweb_local.py` first."
	),
)
def test_integration_dp2_tp4_fineweb_local():
	"""End-to-end: fake 8 CPU devs, mesh=(dp=2,tp=4), data=(dp,), real FineWeb.

	Asserts the three spec properties on real jax.Array shards:
	  (a) tp replicas at the same dp_rank see *identical* rows
	  (b) different dp_ranks see *different* rows
	  (c) global shape and total tokens match make_global_shape's prediction
	"""
	from jax.sharding import Mesh
	import datasets
	from transformers import AutoTokenizer

	from ueaj.data.distributed_loader import (
		distributed_batch_iterator,
		make_global_shape,
	)

	devs = np.asarray(jax.devices()[:8]).reshape(2, 4)
	mesh = Mesh(devs, axis_names=("dp", "tp"))

	# Stream from the local mirror; need num_shards >= DP=2 so .shard() can split.
	hf_ds = datasets.load_dataset(
		"parquet",
		data_files=_LOCAL_SHARDS,
		split="train",
		streaming=True,
	)

	# Match the production tokenizer; pad_token_id derived from the tokenizer.
	tok = AutoTokenizer.from_pretrained("moondream/starmie-v1")
	pad_id = tok.pad_token_id if tok.pad_token_id is not None else (tok.eos_token_id or 0)

	with mesh:
		gen = distributed_batch_iterator(
			hf_ds, tok, mesh, ("dp",),
			batch_size_per_data_rank=1,
			seq_len=32,
			pad_token_id=pad_id,
		)

		expected_shape = make_global_shape(mesh, ("dp",), 1, 32)
		seen = []
		for i, (tokens, doc_ids) in enumerate(gen):
			if i >= 4:
				break
			# (c) shape
			assert tokens.shape == expected_shape, (
				f"expected {expected_shape}, got {tokens.shape}"
			)

			# Group addressable shards by (dp_rank, tp_rank).
			by_dp: dict[int, dict[int, np.ndarray]] = {}
			for shard in tokens.addressable_shards:
				idx = np.argwhere(devs == shard.device)
				assert idx.shape == (1, 2)
				dp_rank, tp_rank = int(idx[0, 0]), int(idx[0, 1])
				by_dp.setdefault(dp_rank, {})[tp_rank] = np.asarray(shard.data)

			# (a) tp-replication: every tp at the same dp must be identical.
			for dp_rank, by_tp in by_dp.items():
				ref = next(iter(by_tp.values()))
				for tp_rank, arr in by_tp.items():
					assert np.array_equal(arr, ref), (
						f"tp replication broken at dp={dp_rank} tp={tp_rank}"
					)

			# (b) cross-dp difference (synthetic stream is unique-per-doc).
			if 0 in by_dp and 1 in by_dp:
				row0 = next(iter(by_dp[0].values()))
				row1 = next(iter(by_dp[1].values()))
				assert not np.array_equal(row0, row1), (
					"dp=0 and dp=1 should see different docs"
				)

			seen.append(tokens.shape)

		# We asked for 4 batches; the synthetic stream is long enough.
		assert len(seen) == 4


# ---------------------------------------------------------------------------
# Layer 3: true multi-process test. Spawns N python subprocesses, each calls
# jax.distributed.initialize() over loopback. This exercises the actual
# cross-process synchronization in make_array_from_process_local_data, which
# Layer 2 (single-process w/ fake CPU devices) cannot.
# ---------------------------------------------------------------------------

import json
import socket
import subprocess
import sys
import tempfile
from pathlib import Path


def _free_port() -> int:
	with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
		s.bind(("127.0.0.1", 0))
		return s.getsockname()[1]


@pytest.mark.skipif(
	not _LOCAL_SHARDS,
	reason=(
		"Layer 3 needs the local FineWeb mirror. "
		"Run `.venv/bin/python scripts/download_fineweb_local.py` first."
	),
)
def test_distributed_multiprocess_real():
	"""Spawn 2 processes, each owning 4 CPU 'devices'. Mesh = (dp=2, tp=4),
	data_axes=('dp',). The spec says:

	  * proc 0 owns row 0 (dp=0, tp=0..3) -> 1 loader slot at (dp=0, tp=0),
	    so it streams data-rank 0.
	  * proc 1 owns row 1 (dp=1, tp=0..3) -> 1 loader slot at (dp=1, tp=0),
	    so it streams data-rank 1.

	The collective ``make_array_from_process_local_data`` is what carries each
	process's local slice across the cluster; this test verifies the visible
	addressable shards on each side match the spec.
	"""
	repo_root = Path(__file__).resolve().parents[1]
	worker = repo_root / "scripts" / "_dist_loader_worker.py"
	assert worker.exists()

	num_processes = 2
	dp, tp = 2, 4
	local_devs = tp  # each process gets one row of the mesh
	port = _free_port()
	coordinator = f"127.0.0.1:{port}"

	with tempfile.TemporaryDirectory() as tmpdir:
		tmp = Path(tmpdir)
		report_paths = [tmp / f"report_{i}.json" for i in range(num_processes)]
		log_paths = [tmp / f"worker_{i}.log" for i in range(num_processes)]
		procs = []
		log_files = []
		for i in range(num_processes):
			env = os.environ.copy()
			env.update({
				"JAX_PLATFORMS": "cpu",
				"XLA_FLAGS": f"--xla_force_host_platform_device_count={local_devs}",
				"JAX_NUM_PROCESSES": str(num_processes),
				"JAX_PROCESS_ID": str(i),
				"JAX_COORDINATOR_ADDRESS": coordinator,
				"DP": str(dp),
				"TP": str(tp),
				"SEQ_LEN": "32",
				"BATCH_PER_DP": "1",
				"NUM_BATCHES": "3",
				"FINEWEB_GLOB": _LOCAL_FINEWEB_GLOB,
				"WORKER_REPORT_PATH": str(report_paths[i]),
				"PYTHONPATH": str(repo_root),
			})
			# Redirect output to a file so we don't deadlock on pipe buffers
			# while the parent waits — workers print plenty during init.
			log_f = open(log_paths[i], "wb")
			log_files.append(log_f)
			procs.append(subprocess.Popen(
				[sys.executable, str(worker)],
				env=env,
				stdout=log_f,
				stderr=subprocess.STDOUT,
			))

		# Poll for both report files. Workers may hang on JAX-distributed
		# shutdown after a clean run on CPU loopback, so we kill them as soon
		# as the report is on disk (ok=True or not) and rely on the report
		# contents for correctness.
		import time
		deadline = time.time() + 120.0
		def _try_load(p: Path) -> dict | None:
			if not p.exists() or p.stat().st_size == 0:
				return None
			try:
				return json.loads(p.read_text())
			except json.JSONDecodeError:
				return None
		reports: list[dict | None] = [None, None]
		while time.time() < deadline:
			for i in range(num_processes):
				if reports[i] is None:
					reports[i] = _try_load(report_paths[i])
			# Also surface early exit (e.g. exception during init).
			for i, p in enumerate(procs):
				if p.poll() is not None and reports[i] is None:
					reports[i] = _try_load(report_paths[i])
			if all(r is not None for r in reports):
				break
			time.sleep(0.5)

		for p in procs:
			if p.poll() is None:
				p.kill()
		for p in procs:
			try: p.wait(timeout=5)
			except subprocess.TimeoutExpired: pass
		for f in log_files: f.close()

		for i, r in enumerate(reports):
			if r is None:
				pytest.fail(
					f"worker {i} produced no report.\nlog tail:\n"
					+ log_paths[i].read_text()[-2000:]
				)
			if not r.get("ok"):
				pytest.fail(
					f"worker {i} reported failure: {r.get('error')!r}\n"
					f"traceback:\n{r.get('traceback', '<none>')}"
				)

	# (1) Each process loads exactly one slot at its own dp_rank.
	for i, r in enumerate(reports):
		assert r["ok"], r.get("traceback", r)
		assert r["process_index"] == i
		assert r["process_count"] == num_processes
		assert r["local_devices"] == local_devs
		assert r["global_devices"] == dp * tp
		assert r["loader_ranks"] == [i], (
			f"proc {i} expected loader_ranks=[{i}], got {r['loader_ranks']}"
		)
		assert r["num_slots"] == 1
		assert r["total_data"] == dp
		assert r["expected_global_shape"] == [dp * 1, 32]

	# (2) For each batch:
	#   - within a single process, all addressable (dp,tp) shards have the
	#     same fingerprint (they all hold the row for that process's dp_rank).
	#   - across processes, the row for dp=0 (seen by proc 0) differs from
	#     the row for dp=1 (seen by proc 1) -- distinct streams.
	num_batches = 3
	assert all(len(r["batches"]) == num_batches for r in reports)
	for b in range(num_batches):
		fps_per_proc = []
		for i, r in enumerate(reports):
			b_shards = r["batches"][b]
			# Proc i should only see (dp=i, tp=0..3) shards.
			expected_keys = {f"{i},{t}" for t in range(tp)}
			assert set(b_shards.keys()) == expected_keys, (
				f"proc {i} batch {b}: shards {set(b_shards.keys())} != {expected_keys}"
			)
			fingerprints = list(b_shards.values())
			ref = fingerprints[0]
			for fp in fingerprints[1:]:
				assert fp == ref, (
					f"tp-replication broken on proc {i} batch {b}"
				)
			fps_per_proc.append(ref)
		# Cross-process: proc 0's row != proc 1's row (different shards of FineWeb).
		assert fps_per_proc[0] != fps_per_proc[1], (
			f"proc 0 and proc 1 saw identical tokens at batch {b}; "
			f"shard split is broken (both processes streaming the same data)"
		)
