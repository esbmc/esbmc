# The annotation a later binding carries can type a global differently
# from the binding itself, so a read converted before it is refused
# (#8369).
def f() -> str:
    return s


s = "hello"
assert f() == "hello"
