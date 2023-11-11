def zero(n:int) -> int:
    if (n>0):
        return zero(n-1)
    return n

result:int = zero(5)