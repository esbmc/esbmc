# Regression for GitHub #4790: list(d) / sorted(d) over a dict were lowered by
# relabeling the dict struct as a list, giving a wrong length and an unsound
# subscript that read a wrong deterministic value. They are now routed through
# d.keys(), the correctly typed dict-keys list.

# list(dict) -> keys in insertion order
d = {7: 70, 3: 30}
ks = list(d)
assert len(ks) == 2
assert ks[0] == 7
assert ks[1] == 3

# sorted(dict) -> sorted list of keys
s = sorted(d)
assert s[0] == 3
assert s[1] == 7

# iteration over list(dict)
total = 0
for k in list(d):
    total += k
assert total == 10

# lowercase dict[...] annotation is recognised too
e: dict[int, int] = {2: 20, 1: 10}
se = sorted(e)
assert se[0] == 1
