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

def ceil(x:float) -> int:
    return 0

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

def arctan(x:float) -> float:
    return 0.2

def dot(a:float, b:float) -> float:
    return 0.0
