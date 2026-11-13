def pick(w: str, k: int) -> str:
    return w[::k]


r = pick("abcdef", -2)
assert r == "fdb"
caught = 0
try:
    pick("abcdef", 0)
except ValueError:
    caught = 1
assert caught == 1
