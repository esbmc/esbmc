a = nondet_int()
if a > 0:
    s = "P"
else:
    s = "N"
if a > 0:
    w = "pos"
else:
    w = "negative"
# A failing assert ends its path, so each check runs on a path of its own.
k = nondet_int()
v = "N"
i = 0
while i < 2:
    if i == 1 and k == 0:
        assert v == "N"
    if a > 0:
        v = "P"
    i += 1
if k == 1:
    assert s == "N"
elif k == 2:
    assert len(w) == 8
elif k == 3:
    assert "N" in s
elif k == 4:
    assert f"<{s}>" == "<N>"
