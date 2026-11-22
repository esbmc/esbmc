words = ["a", "b"]
n = 0
while words and (w := words.pop()):
    n += 1
assert n == 2
assert w == "b"
