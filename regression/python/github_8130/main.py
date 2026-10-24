# #8130: replacing an inner list through a parameter must reach list ==.
def put(x: list) -> None:
    x[0] = [5]

m: list = [[1]]
n: list = [[5]]
put(m)
assert m == n
