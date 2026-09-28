# Scalar variant of the nested rebind: proves the write counter, not
# container mutation, is the mechanism (GitHub #5955).
def t() -> bool:
    return True
x = 1
if t():
    x = 2
assert int(x) == 1
