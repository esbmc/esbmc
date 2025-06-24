# Stubs for type inference.
def array(l: list[Any]) -> list[Any]:
    return l

def zeros(shape:int) -> list[float] :
    l:list[float] = [0.0]
    return l

def ones(shape:int) -> list[float] :
    l:list[float] = [1.0]
    return l

def add(a:int, b:int) -> float:
    x:float = a + b
    return x

def subtract(a:int, b:int) -> float:
    x:float = a - b
    return x

def multiply(a:int, b:int) -> float:
    x:float = a * b
    return x

def divide(a:int, b:int) -> float:
    x:float = a / b
    return x

def power(a:int, b:int) -> float:
    x:float = 42.0
    return x
<<<<<<< HEAD
=======

def ceil(x:float) -> int:
    return 0
<<<<<<< HEAD
>>>>>>> f59fd87f9 ([python-frontend] Reuse C libm models for NumPy math functions (#2395))
=======

def floor(x:float) -> int:
    return 0

def fabs(x:float) -> float:
    return 1.0

def sqrt(x:float) -> float:
    return 0.2

def fmin(x:float, y:float) -> float:
    return 0.2

def fmax(x:float, y:float) -> float:
    return 0.2

def trunc(x:float) -> float:
    return 0.2

def round(x:float) -> float:
    return 0.2

def copysign(x:float, y:float) -> float:
    return 0.2

def sin(x:float) -> float:
    return 0.2

def cos(x:float) -> float:
    return 0.2

def arccos(x:float) -> float:
    return 0.2

def exp(x:float) -> float:
    return 0.2

def fmod(x:float) -> float:
    return 0.2
<<<<<<< HEAD
>>>>>>> 2e5aef15f ([python] Expand numpy math models (#2407))
=======

def arctan(x:float) -> float:
    return 0.2
<<<<<<< HEAD
>>>>>>> f86e3ad14 ([python] added test cases for numpy math functions (#2437))
=======

def dot(a:float, b:float) -> float:
    return 0.0
<<<<<<< HEAD
>>>>>>> bea8feb3b ([python] Handling NumPy dot product (#2460))
=======

def matmul(a:float, b:float) -> float:
    return 0.0

def transpose(a:float, b:float) -> float:
    return 0.0

>>>>>>> 26666f4c8 ([python] Handling numpy functions (#2474))
