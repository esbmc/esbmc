class A(Exception):
    pass


class B(A):
    pass


caught = False
try:
    raise B()
except A:
    caught = True

assert caught
