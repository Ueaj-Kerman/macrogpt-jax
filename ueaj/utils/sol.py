"""Speed-of-Light (SOL) and MFU helpers.

Two layers:
  1. Static: ask XLA's cost analysis for theoretical FLOPs / bytes accessed of a
     compiled function. Cheap, run-once per compile.
  2. Dynamic: divide static FLOPs by measured wall time to get achieved
     TFLOPs/s, and divide by the chip's peak to get % of SOL.

For per-stage SOL (forward / backward / optimizer_update / stats), the named
scopes added in `ueaj.train.training_utils` show up in `jax.profiler.trace`
output. Parsing that trace offline is left to a separate tool (utilyze for
hardware counters; xprof for HLO time). This module is intentionally a thin
helper, not a parser.
"""

from __future__ import annotations

import dataclasses
import json
from typing import Any, Mapping, Optional

import jax


# Theoretical peaks for accelerators we care about. bf16 dense matmul TFLOPs and
# HBM bandwidth in GB/s. Numbers are vendor-published peaks, not measured.
PEAK_SPECS: dict[str, dict[str, float]] = {
    "H100":         {"tflops_bf16": 989.0,  "hbm_gbps": 3350.0},
    "H100_SXM":     {"tflops_bf16": 989.0,  "hbm_gbps": 3350.0},
    "H100_PCIe":    {"tflops_bf16": 756.0,  "hbm_gbps": 2000.0},
    "B200":         {"tflops_bf16": 2250.0, "hbm_gbps": 8000.0},
    "GB200":        {"tflops_bf16": 2500.0, "hbm_gbps": 8000.0},
    "A100":         {"tflops_bf16": 312.0,  "hbm_gbps": 2039.0},
    "RTX_5090":     {"tflops_bf16": 419.0,  "hbm_gbps": 1792.0},
    "RTX_4090":     {"tflops_bf16": 165.0,  "hbm_gbps": 1008.0},
}


@dataclasses.dataclass(frozen=True)
class StepCost:
    flops: float
    bytes_accessed: float

    def achieved_tflops_per_s(self, wall_time_s: float) -> float:
        return self.flops / wall_time_s / 1e12

    def achieved_hbm_gbps(self, wall_time_s: float) -> float:
        return self.bytes_accessed / wall_time_s / 1e9

    def mfu(self, wall_time_s: float, peak_tflops: float) -> float:
        return self.achieved_tflops_per_s(wall_time_s) / peak_tflops

    def mbu(self, wall_time_s: float, peak_gbps: float) -> float:
        return self.achieved_hbm_gbps(wall_time_s) / peak_gbps


def cost_of_compiled(compiled: Any) -> Optional[StepCost]:
    """Pull FLOPs and bytes_accessed from a compiled JAX function.

    `compiled` is the output of `jax.jit(fn).lower(...).compile()` or anything
    with a `.cost_analysis()` method. Returns None if cost analysis is
    unavailable on this backend (e.g. CPU often is).
    """
    try:
        analysis = compiled.cost_analysis()
    except Exception:
        return None
    if analysis is None:
        return None
    if isinstance(analysis, list):
        if not analysis:
            return None
        analysis = analysis[0]
    flops = float(analysis.get("flops", 0.0) or 0.0)
    bytes_accessed = float(analysis.get("bytes accessed", 0.0) or 0.0)
    if flops == 0.0 and bytes_accessed == 0.0:
        return None
    return StepCost(flops=flops, bytes_accessed=bytes_accessed)


def detect_chip() -> str | None:
    """Best-effort GPU detection. Returns a key into PEAK_SPECS or None."""
    try:
        devs = jax.devices()
    except Exception:
        return None
    for d in devs:
        if d.platform != "gpu":
            continue
        kind = (getattr(d, "device_kind", "") or "").lower()
        if "h100" in kind:
            if "pcie" in kind:
                return "H100_PCIe"
            return "H100"
        if "b200" in kind:
            return "B200"
        if "gb200" in kind:
            return "GB200"
        if "a100" in kind:
            return "A100"
        if "5090" in kind:
            return "RTX_5090"
        if "4090" in kind:
            return "RTX_4090"
    return None


def report(
    cost: StepCost,
    wall_time_s: float,
    chip: str | None = None,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, float]:
    """Format a one-step SOL report. Returns a flat dict for easy logging."""
    chip = chip or detect_chip()
    out: dict[str, Any] = {
        "wall_time_s": wall_time_s,
        "flops": cost.flops,
        "bytes_accessed": cost.bytes_accessed,
        "tflops_per_s": cost.achieved_tflops_per_s(wall_time_s),
        "hbm_gbps": cost.achieved_hbm_gbps(wall_time_s),
        "chip": chip,
    }
    if chip and chip in PEAK_SPECS:
        peak = PEAK_SPECS[chip]
        out["mfu"] = cost.mfu(wall_time_s, peak["tflops_bf16"])
        out["mbu"] = cost.mbu(wall_time_s, peak["hbm_gbps"])
        out["peak_tflops"] = peak["tflops_bf16"]
        out["peak_hbm_gbps"] = peak["hbm_gbps"]
    if extra:
        out.update(extra)
    return out


def format_line(rep: Mapping[str, Any]) -> str:
    """Compact one-liner for stdout."""
    parts = [f"{rep['tflops_per_s']:.1f} TFLOP/s"]
    if "mfu" in rep:
        parts.append(f"MFU={rep['mfu']*100:.1f}%")
    parts.append(f"{rep['hbm_gbps']:.0f} GB/s")
    if "mbu" in rep:
        parts.append(f"MBU={rep['mbu']*100:.1f}%")
    parts.append(f"step={rep['wall_time_s']*1000:.1f}ms")
    if rep.get("chip"):
        parts.append(f"({rep['chip']})")
    return " | ".join(parts)


def save(rep: Mapping[str, Any], path: str) -> None:
    """Dump a report to JSON for offline diffing across runs."""
    with open(path, "w") as f:
        json.dump(dict(rep), f, indent=2, default=str)
