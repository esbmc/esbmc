# ord() of a non-string raises TypeError. Inside a ternary condition the
# exception must propagate, not crash goto_check with a null if-condition.
def f(x: int) -> bool:
    return True if 97 <= ord(x) else False


if __name__ == "__main__":
    # f(5) raises TypeError inside the ternary condition before the assert is
    # ever evaluated; the verdict is an uncaught exception, not an assert fail.
    assert f(5) == True
