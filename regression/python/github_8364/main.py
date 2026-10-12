# KNOWNBUG until #8384: a method call on a list element is not dispatched.
class P:
    def size(self, k: int) -> int:
        return k

def size() -> int:
    return 0

ps = [P()]
assert ps[0].size(2) == 2
