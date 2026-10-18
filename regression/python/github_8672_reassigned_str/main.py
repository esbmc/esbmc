x = 2
if nondet_bool():
    x = 1
s = str(x)
assert s == "2" or s == "1"
assert f"<{x}>" == "<2>" or f"<{x}>" == "<1>"
