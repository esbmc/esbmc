words = ["a", "", "b"]
out: list[str] = []
while words and (w := words.pop()):
    out.append(w)
assert out == ["b"]
assert len(words) == 1

xs: list[str] = []
while xs and (v := xs.pop()):
    assert False
