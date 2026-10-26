# The class is called before its name is rebound, so that call builds an
# instance.
class Count:

    def __init__(self, n: int):
        self.n = n


def pair(n: int) -> list[int]:
    return [7, 8]


c = Count(1)
Count = pair
assert c.n == 1
