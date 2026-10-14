# A helper nested in __init__ is distinct from a method with the same name (#7947).
class K:
    def __init__(self):
        def helper() -> int:
            t = "ab"
            t = "xyz"
            return len(t)

        self.n = helper()

    def helper(self) -> int:
        t = "abcdef"
        return len(t)


k = K()
assert k.n == 6
