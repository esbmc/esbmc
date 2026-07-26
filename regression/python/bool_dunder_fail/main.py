# __bool__ returns False, so bool(C()) is False, not True.
class C:
    def __bool__(self):
        return False


assert bool(C()) == True
