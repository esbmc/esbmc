var = 8
print(var)

var = 3.142
assert var > 0

# var is rebound int -> float -> str. The straight-line retyping fix (#4774)
# preserves the real string value, so asserting a different string must FAIL.
var = 'Python in easy steps'
assert var == 'C in easy steps'
