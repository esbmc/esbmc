class K:
    def __init__(self) -> None:
        self.x = 1


s = str(K())
assert s.startswith("<__main__.K object at 0x")
assert s.endswith(">")
assert len(s) > 25

k = K()
assert str(k)[0] == "<"
k = None
assert str(k) == "None"
