from stubs import *

def f(a: A) -> int:
    return a[2:5, 10:20]

def g(b: B) -> int:
    return b[1:2, 2:5, 0:64, 0:128]

x: int = f(A())
y: int = g(B())
