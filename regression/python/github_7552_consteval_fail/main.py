# Constant evaluation folded these by UTF-8 bytes and proved them (#7552).
def tail(t: str) -> str:
    return t[1:]


s = "é"
if nondet_bool():
    assert len(s) == 2
else:
    assert len(tail("éa")) == 2
