# GitHub #9015: the call must be evaluated, so this assertion fails.
class R:
    def m(self):
        return 1


assert not isinstance(R().m(), int)
