# Calling the result of a call through a variable that holds a function
# (#8961).
def h():
    return 2


def get():
    return h


f = get
assert f()() == 2
