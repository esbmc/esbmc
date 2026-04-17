def step_until_two():
    i = 0
    while i < 2:
        nextnode = i + 1
        i = nextnode
    return i


assert step_until_two() == 2
