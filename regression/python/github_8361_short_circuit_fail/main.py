words = ["a", "", "b"]
n = 0
while words and (w := words.pop()):
    n += 1
assert w == "b"
