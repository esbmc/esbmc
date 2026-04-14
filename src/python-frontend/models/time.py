<<<<<<< HEAD
_esbmc_time_now: float = 0.0
=======
# Operational model for the time module


def sleep(seconds) -> None:
    """Sleep does nothing for verification."""
    return None
>>>>>>> 49ac0663be ([python] capitalize)


def time() -> float:
    global _esbmc_time_now
    current: float = _esbmc_time_now
    _esbmc_time_now = _esbmc_time_now + 1.0
    return current


def sleep(seconds: float) -> None:
    assert seconds >= 0.0
