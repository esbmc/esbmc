def f() -> str:
    return "a"


x = [f(), 1]
for i in range(1):
    assert x[i] == "a"
