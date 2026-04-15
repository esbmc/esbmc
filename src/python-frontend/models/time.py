_esbmc_time_now: float = 0.0


def time() -> float:
    global _esbmc_time_now
    current: float = _esbmc_time_now
    _esbmc_time_now = _esbmc_time_now + 1.0
    return current


def sleep(seconds: float) -> None:
    assert seconds >= 0.0
