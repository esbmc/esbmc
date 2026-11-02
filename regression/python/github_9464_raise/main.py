# The class of `other` is not known statically; the text of the message is.
class V:
    def __add__(self, other):
        if not isinstance(other, V):
            raise ValueError("Can't add V to " + str(type(other)))
        return self


v = V()
c = nondet_bool()
raised = False
try:
    if c:
        v + 1
    else:
        v + V()
except ValueError:
    raised = True
assert raised == c
