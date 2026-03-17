# Unpacking without star from a list variable
def head_tail(lst: list[int]):
    first, second, third = lst
    return first + second + third

assert head_tail([10, 20, 30]) == 60
