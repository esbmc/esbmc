# len() of a call that returns None instead of an instance (#10505)
class C:

    def __len__(self) -> int:
        return 1


def make(b: bool) -> C:
    if b:
        return C()
    return None


assert len(make(True)) == 1
assert len(make(False)) == 1
