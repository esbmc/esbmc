# The same with the exception class nested in another class.
class A:

    class B(Exception):

        def __init__(self):
            pass


a = A.B()
if nondet_bool():
    a.code = 1
y = a.code
