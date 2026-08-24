def gen():
    yield 1
    yield 2


class Acc:
    def __init__(self) -> None:
        self.t: int = 0


def main():
    t: int = 0

    for v in range(1, 4):
        t = t + v
    assert t == 6

    for v in [10, 20]:
        t = t + v
    assert t == 36

    xs = [1, 2, 3]
    for v in xs:
        t = t + v
    assert t == 42

    for v in (4, 5):
        t = t + v
    assert t == 51

    for ch in "abc":
        t = t + 1
    assert t == 54

    d = {"a": 1, "b": 2}
    for k in d.keys():
        t = t + d[k]
    assert t == 57

    for i, v in enumerate([100, 200]):
        t = t + i + v
    assert t == 358

    for a, b in zip([1, 2], [10, 20]):
        t = t + a * b
    assert t == 408

    for v in reversed([1, 2, 3]):
        t = t * 10 + v
    assert t == 408321

    for v in gen():
        t = t + v
    assert t == 408324

    for a in [1, 2]:
        for b in [10, 20]:
            t = t + a * b
    assert t == 408414

    for v in [7]:
        t = t + v
    else:
        t = t + 1
    assert t == 4242


main()
