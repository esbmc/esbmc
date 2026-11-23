def g(s: str) -> int:
    return len(s)


assert g("é") == 1
assert g("aé\U0001F600") == 3
t = "é" if nondet_bool() else "aé"
assert g(t) <= 2
