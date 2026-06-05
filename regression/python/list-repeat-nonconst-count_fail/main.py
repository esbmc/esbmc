# Negative variant: a wrong size for a runtime-count repeat must not verify.
def f(n: int) -> int:
    a = [0] * (n + 1)
    return len(a)


if __name__ == "__main__":
    # f(5) builds 6 elements; asserting 1 (the old buggy size) must fail.
    assert f(5) == 1
