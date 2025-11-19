
def hanoi(height, start=1, end=3):
    steps = []
    if height > 0:
        helper = ({1, 2, 3} - {start} - {end}).pop()
        steps.extend(hanoi(height - 1, start, helper))
        steps.append((start, end))
        steps.extend(hanoi(height - 1, helper, end))

    return steps

assert hanoi(0, 1, 3) == []
assert hanoi(1, 1, 3) == [(1, 3)]
assert hanoi(2, 1, 3) == [(1, 2), (1, 3), (2, 3)]
assert hanoi(3, 1, 3) == [(1, 3), (1, 2), (3, 2), (1, 3), (2, 1), (2, 3), (1, 3)]
assert hanoi(4, 1, 3) == [(1, 2), (1, 3), (2, 3), (1, 2), (3, 1), (3, 2), (1, 2), (1, 3), (2, 3), (2, 1), (3, 1), (2, 3), (1, 2), (1, 3), (2, 3)]
assert hanoi(2, 1, 2) == [(1, 3), (1, 2), (3, 2)]
assert hanoi(2, 1, 1) == [(1, 2), (1, 1), (2, 1)]
assert hanoi(2, 3, 1) == [(3, 2), (3, 1), (2, 1)]