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
    y:int = int(x)
    if x > 0 and x != y:
      return y + 1
    return y

# def ceil_list(l: list[float]) -> None:
#     i: int = 0
#     while i < len(l):
#         l[i] = ceil(l[i])
#         i = i + 1
        
