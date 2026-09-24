# A class defined after the len() call still counts, so A.__len__ is not
# dispatched on a B element (#7966).
class A:
    def __len__(self) -> int:
        return 1
def f(xs: list[A]) -> int:
    return len(xs[0])
class B(A):
    def __len__(self) -> int:
        return 2
assert f([B()]) == 1
