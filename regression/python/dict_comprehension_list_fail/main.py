# Negative companion: the comprehension is built correctly, but the assertion
# is wrong, so verification must FAIL (guards against a vacuous/over-strong fix
# that would make every lookup succeed).
nums = [1, 2, 3]
squares = {n: n * n for n in nums}
assert squares[2] == 5
