# Control for github_7397: the same program without the colliding import.


class Shape:
    def __init__(self) -> None:
        self.edges: int = 3

    def total(self) -> int:
        return self.edges


s = Shape()
assert s.total() == 3
