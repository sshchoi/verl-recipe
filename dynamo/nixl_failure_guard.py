"""Interim rollout safeguard based on every P/D worker's published NIXL counters."""

import math
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.request import urlopen

_SAMPLE = re.compile(r"^(vllm:nixl_num_failed_transfers_(?:total|created)(?:\{.*\})?)\s+(\S+)")


class NixlFailureGuard:
    """Latch on any observed failure or loss of telemetry."""

    def __init__(self, endpoint_dir, expected_workers, timeout=5.0):
        self.endpoint_dir = Path(endpoint_dir)
        self.expected_workers = int(expected_workers)
        self.timeout = float(timeout)
        if self.expected_workers < 1 or not math.isfinite(self.timeout) or self.timeout <= 0:
            raise ValueError("NIXL guard needs a positive expected_workers and timeout")
        self._baseline = None
        self._failure = None

    def _scrape(self, endpoint):
        with urlopen(f"http://{endpoint}/metrics", timeout=self.timeout) as response:
            text = response.read().decode("utf-8")
        samples = {}
        for line in text.splitlines():
            match = _SAMPLE.match(line)
            if match:
                key, raw_value = match.groups()
                value = float(raw_value)
                if not math.isfinite(value) or value < 0 or key in samples:
                    raise ValueError(f"Invalid NIXL metric from {endpoint}: {line}")
                samples[key] = value
        totals = {k: v for k, v in samples.items() if k.split("{", 1)[0].endswith("_total")}
        if not totals:
            raise ValueError(f"NIXL failure counter missing from {endpoint}")
        if any(totals.values()):
            raise ValueError(f"NIXL transfer failure on {endpoint}: {totals}")
        return endpoint, samples

    def check(self):
        if self._failure is not None:
            raise RuntimeError(self._failure)
        try:
            endpoints = sorted(
                {
                    line.strip()
                    for path in self.endpoint_dir.glob("*.pd-endpoints")
                    for line in path.read_text().splitlines()
                    if line.strip()
                }
            )
            if len(endpoints) != self.expected_workers:
                raise ValueError(f"Expected {self.expected_workers} P/D workers, found {len(endpoints)}")
            with ThreadPoolExecutor(max_workers=min(16, len(endpoints))) as pool:
                samples = dict(pool.map(self._scrape, endpoints))
            if self._baseline is not None and samples != self._baseline:
                raise ValueError("P/D metric targets, series, or counter creation times changed")
            self._baseline = samples
        except Exception as error:
            self._failure = (
                f"NIXL rollout guard stopped sampling: {error}. "
                "Discard trajectories from the failure window and restart the affected worker; "
                "do not resume this rollout step."
            )
            raise RuntimeError(self._failure) from error
