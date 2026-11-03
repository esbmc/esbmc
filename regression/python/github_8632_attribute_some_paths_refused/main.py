# code exists only when the branch is taken, so the read raises
# AttributeError on the other path; conversion refuses the store.
class A:
    pass


a = A()
if nondet_bool():
    a.code = 1
y = a.code
