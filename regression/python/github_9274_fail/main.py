def f() -> str:
    return "a"


x = [f(), f()]
for i in range(2):
    assert x[i] == "b"
