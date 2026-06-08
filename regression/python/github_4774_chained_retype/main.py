# Chained straight-line retyping (#4774). A module global is rebound across
# several incompatible types in sequence. Each reassignment must adopt the new
# value and type, so the final assertions hold only if every link in the chain
# is tracked (not just the first).
v = 8
v = 'hello'
assert v == 'hello'

v = 42
assert v == 42

v = 'world'
assert v == 'world'

v = 3.5
assert v > 3.0
