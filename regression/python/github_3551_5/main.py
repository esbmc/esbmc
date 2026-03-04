def f() -> float:
    diff: float = -1.0
    if diff < 0.0:
        diff = -diff
    return diff


if __name__ == "__main__":
    assert abs(f() - 1.0) < 1e-6
