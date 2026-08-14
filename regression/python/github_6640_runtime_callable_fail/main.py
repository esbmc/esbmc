# The call must dispatch to the callable that was actually selected, not to
# whichever name the frontend saw first: pick(False) is century, so expecting
# inc's result has to be refuted (#6640).
def inc(m: int) -> int:
    return m + 1


def century(m: int) -> int:
    return m + 100


def pick(c: bool):
    return inc if c else century


chosen = pick(False)
assert chosen(1) == 2
