def pick(w: str, k: int) -> str:
    return w[::k]


r = pick("abcdef", -2)
assert r == "abcdef"
