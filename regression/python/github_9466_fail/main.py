# github issue #9466: the later pair for a repeated key wins
b = [1, 2]
e = [3, 4]
m = {x: y for x in b for y in e}
assert m[1] == 3
