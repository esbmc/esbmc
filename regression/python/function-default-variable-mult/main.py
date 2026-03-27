x = 1
y = 2


def return_prod(a: int = x, b: int = y) -> int:
    return a * b


assert return_prod(4, 4) == 16
assert return_prod(3) == 6
assert return_prod(a=3, b=4) == 12

x = 2

assert return_prod(4, 4) == 16
assert return_prod(3) == 6
assert return_prod(a=3, b=4) == 12

y = 3

assert return_prod(4, 4) == 16
assert return_prod(3) == 6
assert return_prod(a=3, b=4) == 12
