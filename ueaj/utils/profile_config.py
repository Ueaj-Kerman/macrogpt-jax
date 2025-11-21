"""
Profiling configuration via environment variables.

Environment Variables:
    PROFILE_ENABLED: Set to '1' to enable profiling (default: '0')
    PROFILE_DIR: Directory for profile outputs (default: './profiles')
    PROFILE_START_STEP: Step to start profiling (default: 100)
    PROFILE_INTERVAL: Profile every N steps (default: 1000, 0 = only once)
    PROFILE_DURATION: Number of steps to profile (default: 1)
    PROFILE_MODE: 'tensorboard' or 'perfetto' (default: 'tensorboard')

Example:
    PROFILE_ENABLED=1 PROFILE_START_STEP=50 .venv/bin/python -m ueaj.train.train
"""

import os
from contextlib import contextmanager
from typing import Optional
from ueaj.utils.profiling import profile_trace


class ProfilingConfig:
    """Global profiling configuration from environment variables."""

    def __init__(self):
        self.enabled = os.getenv('PROFILE_ENABLED', '0') == '1'
        self.profile_dir = os.getenv('PROFILE_DIR', './profiles')
        self.start_step = int(os.getenv('PROFILE_START_STEP', '100'))
        self.interval = int(os.getenv('PROFILE_INTERVAL', '1000'))
        self.duration = int(os.getenv('PROFILE_DURATION', '1'))
        self.mode = os.getenv('PROFILE_MODE', 'tensorboard')  # or 'perfetto'

        # Track profiling state
        self._profiling_steps_remaining = 0
        self._last_profiled_step = -1

        if self.enabled:
            print("=" * 60)
            print("📊 PROFILING ENABLED")
            print("=" * 60)
            print(f"  Profile Dir:    {self.profile_dir}")
            print(f"  Start Step:     {self.start_step}")
            print(f"  Interval:       {self.interval if self.interval > 0 else 'once'}")
            print(f"  Duration:       {self.duration} step(s)")
            print(f"  Mode:           {self.mode}")
            print("=" * 60)

    def should_profile(self, step: int) -> bool:
        """Determine if we should profile this step."""
        if not self.enabled:
            return False

        # Check if we're in the middle of a multi-step profile
        if self._profiling_steps_remaining > 0:
            return True

        # Check if this is a step where we should start profiling
        if step < self.start_step:
            return False

        # One-time profiling (interval == 0)
        if self.interval == 0:
            if self._last_profiled_step == -1 and step >= self.start_step:
                self._profiling_steps_remaining = self.duration - 1
                self._last_profiled_step = step
                return True
            return False

        # Periodic profiling
        if (step - self.start_step) % self.interval == 0:
            self._profiling_steps_remaining = self.duration - 1
            self._last_profiled_step = step
            return True

        return False

    def get_profile_name(self, step: int) -> str:
        """Generate profile name for this step."""
        return f"step_{step:08d}"

    @contextmanager
    def profile_step(self, step: int):
        """
        Context manager for profiling a training step.

        Usage:
            config = ProfilingConfig()
            with config.profile_step(step):
                loss = train_step(...)
                loss.block_until_ready()
        """
        should_profile = self.should_profile(step)

        if should_profile:
            print(f"\n📊 Profiling step {step}...")
            profile_name = self.get_profile_name(step)

            # Decrement remaining steps for multi-step profiles
            if self._profiling_steps_remaining > 0:
                self._profiling_steps_remaining -= 1

            # Choose mode
            create_perfetto = (self.mode == 'perfetto')

            with profile_trace(
                self.profile_dir,
                name=profile_name,
                create_perfetto_link=create_perfetto,
                tensorboard=not create_perfetto
            ):
                yield
            print(f"✓ Profile saved: {self.profile_dir}/{profile_name}")
        else:
            # No profiling, just yield
            yield


# Global instance
_global_config: Optional[ProfilingConfig] = None


def get_profiling_config() -> ProfilingConfig:
    """Get or create global profiling config."""
    global _global_config
    if _global_config is None:
        _global_config = ProfilingConfig()
    return _global_config


@contextmanager
def maybe_profile(step: int):
    """
    Simple context manager that profiles if enabled via env vars.

    Usage in training loop:
        from ueaj.utils import maybe_profile

        for step in range(max_steps):
            with maybe_profile(step):
                loss = train_step(model, batch)
                loss.block_until_ready()
    """
    config = get_profiling_config()
    with config.profile_step(step):
        yield
