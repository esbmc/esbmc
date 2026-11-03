# A tagged return used directly as a binop operand (not through a
# variable first) raises TypeError on its str branch.
def f(x: int):
    if x > 0:
        return x
    return "neg"

y = f(-5) + 1
