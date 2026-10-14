from stubs import *

def f(a: A) -> int:
    return a[2:5, 10:20]

x: int = f(A())
