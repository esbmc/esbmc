# The operand reaches __add__ through a function with two kinds of argument.
class V:
    def __add__(self, other):
        if not isinstance(other, V):
            raise ValueError("Can't add V to " + str(type(other)))
        return self


def add(a, b):
    return a + b


r = add(V(), V())
c = nondet_bool()
raised = False
try:
    if c:
        add(V(), 1)
    else:
        add(V(), V())
except ValueError:
    raised = True
assert raised == c
