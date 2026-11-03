# Issue #9084: an exception raised inside str(f()) is lost.
def label() -> str:
    raise ValueError("x")


s = str(label())
