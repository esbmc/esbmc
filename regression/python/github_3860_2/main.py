# Regression test for GitHub issue #3860 (variant 2):
# Generator with arguments and loop-based yields converted to list.


def count_up(n: int):
    i: int = 0
    while i < n:
        yield i
        i = i + 1


result = list(count_up(3))
assert len(result) == 3
assert result[0] == 0
assert result[1] == 1
assert result[2] == 2
