# #8130: reordering inner lists through a parameter must reach list ==.
def f(x: list) -> None:
    x.reverse()

m: list = [[1], [2]]
n: list = [[1], [2]]
f(m)
assert m == n
