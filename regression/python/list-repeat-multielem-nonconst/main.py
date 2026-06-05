# Known limitation (FUTURE): repeating a MULTI-element source by a runtime count
# currently repeats only the first element. CPython: [1, 2] * 3 == [1,2,1,2,1,2].
# Single-element runtime-count repetition (the issue #5146 case) is fixed; this
# multi-element variant is a pre-existing gap tracked here.
def f(n: int) -> int:
    a = [1, 2] * n
    return len(a)


if __name__ == "__main__":
    assert f(3) == 6
