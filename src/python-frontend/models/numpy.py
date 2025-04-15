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
    x:float = a ** b
    return x

def ceil(x:float) -> int:
    return 0

def floor(x:float) -> int:
    return 0
