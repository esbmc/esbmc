# `words and (w := words.pop())` must not pop an empty list, and the loop
# stops at the falsy element with `w` bound to it.
empty: list[str] = []
while empty and (v := empty.pop()):
    pass
assert len(empty) == 0

words = ["a", "", "b"]
n = 0
while words and (w := words.pop()):
    n += 1
assert n == 1
assert w == ""
assert words == ["a"]
