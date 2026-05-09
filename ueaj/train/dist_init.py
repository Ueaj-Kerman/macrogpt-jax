"""Helper for initializing JAX distributed runtime from environment variables.

Idempotent: a no-op when only a single process is detected, and safe to call
multiple times (subsequent calls are skipped).

Supported env-var conventions (in order of preference):

1. Modal-style explicit vars:
     - JAX_COORDINATOR_ADDRESS  (e.g. "10.0.0.1:1234")
     - JAX_NUM_PROCESSES        (total number of processes across all hosts)
     - JAX_PROCESS_ID           (this process's global rank, 0..N-1)
2. SLURM-style fallbacks (if the above are unset):
     - SLURM_JOB_NODELIST + SLURM_NTASKS + SLURM_PROCID
3. Single-process: if none of the above are set and JAX already reports
   process_count() == 1 nothing happens.
"""

from __future__ import annotations

import os
import socket
from typing import Optional

import jax


_INITIALIZED = False


def _resolve_coordinator() -> Optional[tuple[str, int, int]]:
	"""Return (coordinator_address, num_processes, process_id) or None."""
	addr = os.environ.get("JAX_COORDINATOR_ADDRESS")
	num = os.environ.get("JAX_NUM_PROCESSES")
	pid = os.environ.get("JAX_PROCESS_ID")
	if addr and num and pid:
		return addr, int(num), int(pid)
	# Modal-specific fallback: i6pn private IP for rank 0 + ranks via container env
	num = os.environ.get("MODAL_NUM_PROCESSES") or num
	pid = os.environ.get("MODAL_PROCESS_ID") or pid
	addr = os.environ.get("MODAL_COORDINATOR_ADDRESS") or addr
	if addr and num and pid:
		return addr, int(num), int(pid)
	return None


def maybe_init_distributed(verbose: bool = True) -> bool:
	"""Initialize jax.distributed if multi-process env vars are present.

	Returns True if jax.distributed.initialize was called, False otherwise.
	"""
	global _INITIALIZED
	if _INITIALIZED:
		return True

	resolved = _resolve_coordinator()
	if resolved is None:
		if verbose:
			print(
				f"[dist_init] no multi-process env vars found; "
				f"running single-process (jax.process_count={jax.process_count()})"
			)
		return False

	coordinator_address, num_processes, process_id = resolved

	if num_processes <= 1:
		if verbose:
			print("[dist_init] num_processes <= 1, skipping jax.distributed.initialize")
		return False

	if verbose:
		print(
			f"[dist_init] host={socket.gethostname()} "
			f"initializing jax.distributed: "
			f"coordinator={coordinator_address} "
			f"num_processes={num_processes} process_id={process_id}"
		)

	jax.distributed.initialize(
		coordinator_address=coordinator_address,
		num_processes=num_processes,
		process_id=process_id,
	)
	_INITIALIZED = True

	if verbose:
		print(
			f"[dist_init] ready. process_index={jax.process_index()} "
			f"process_count={jax.process_count()} "
			f"local_devices={len(jax.local_devices())} "
			f"global_devices={jax.device_count()}"
		)
	return True
