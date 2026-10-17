# A method call on a list element dispatches to the element's class (#8384).
class P:
    def size(self, k: int) -> int:
        return k

def size() -> int:
    return 0

ps = [P()]
assert ps[0].size(2) == 2
