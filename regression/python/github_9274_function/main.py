def f(n: int) -> str:
    return "a"


x = [f(1), f(2)]
for i in range(2):
    assert x[i] == "a"
