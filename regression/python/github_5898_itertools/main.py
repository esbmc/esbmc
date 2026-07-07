import itertools


def first_n(iterable, n: int):
    result = list(itertools.islice(iterable, n))
    return result
