def first_positive(x: list[int]) -> bool:
    if len(x) > 0:
        if x[0] > 0:
            return True
    return False


x = nondet_list(2)
first_positive(x)
