"""Run a command while sampling the memory of the job's cgroup.

The cgroup holds every process of the job, so the samples are the job's
memory, not one test's. memory.peak, when the kernel provides it, is the
cgroup's peak since it was created and can include earlier steps.
"""

import os
import subprocess
import threading
import time

CGROUP_ROOT = "/sys/fs/cgroup"


def cgroup_dir():
    """The cgroup v2 directory of this process, or None with no unified hierarchy."""
    try:
        with open("/proc/self/cgroup", encoding="utf-8") as fp:
            for line in fp:
                if line.startswith("0::"):
                    path = os.path.join(CGROUP_ROOT, line[3:].strip().lstrip("/"))
                    return path if os.path.isfile(os.path.join(path, "memory.current")) else None
    except OSError:
        return None
    return None


def _read(directory, name):
    try:
        with open(os.path.join(directory, name), encoding="utf-8") as fp:
            return fp.read().strip()
    except OSError:
        return None


def _events(directory):
    text = _read(directory, "memory.events") or ""
    return {k: int(v) for k, v in (line.split() for line in text.splitlines())}


class _Sampler(threading.Thread):

    def __init__(self, directory, interval):
        super().__init__(daemon=True)
        self.directory, self.interval = directory, interval
        self.samples, self.stop = [], threading.Event()

    def run(self):
        while not self.stop.is_set():
            value = _read(self.directory, "memory.current")
            if value is not None:
                self.samples.append(int(value))
            self.stop.wait(self.interval)


def measure(command, interval=1.0):
    """Run `command`; return its exit code and what was observed."""
    directory = cgroup_dir()
    record = {"command": command, "cgroup": directory}
    sampler = None
    if directory:
        record["memory_start"] = int(_read(directory, "memory.current"))
        record["memory_max_limit"] = _read(directory, "memory.max")
        record["events_before"] = _events(directory)
        sampler = _Sampler(directory, interval)
        sampler.start()
    else:
        record["unavailable"] = "no cgroup v2 memory controller for this process"
    start = time.monotonic()
    exit_code = subprocess.run(command, check=False).returncode
    record["seconds"] = round(time.monotonic() - start, 3)
    record["exit_code"] = exit_code
    if sampler:
        sampler.stop.set()
        sampler.join()
        record["samples"] = len(sampler.samples)
        record["memory_max_sampled"] = max(sampler.samples, default=None)
        peak = _read(directory, "memory.peak")
        record["memory_peak_cgroup"] = int(peak) if peak else None
        record["events_after"] = _events(directory)
        record["oom_kills"] = (record["events_after"].get("oom_kill", 0) -
                               record["events_before"].get("oom_kill", 0))
    return exit_code, record
