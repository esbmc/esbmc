class A(Exception):
    pass


class B(A):
    pass


caught = False
try:
    raise B()
except B:
    caught = True

assert caught
