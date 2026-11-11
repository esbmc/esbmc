# Constant evaluation must not fold a non-ASCII str by its UTF-8 bytes (#7552).
def tail(t: str) -> str:
    return t[1:]


s = "é"
assert len(s) == 1
assert len(tail("éa")) == 1
assert "aé".find("é") + len("ab") == 3
