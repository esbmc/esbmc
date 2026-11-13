# code exists only when the branch is taken, and neither E nor Exception
# binds it, so the read raises AttributeError on the other path.
class E(Exception):
    pass


a = E()
if nondet_bool():
    a.code = 1
y = a.code
