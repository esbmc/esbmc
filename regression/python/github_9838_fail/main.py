# Issue #9838: a missing attribute as an `or` operand of an if test raises.
try:
    from esbmc import nondet_bool
except ImportError:
    def nondet_bool() -> bool:
        return True


class C:
    pass


c = C()
if nondet_bool() or c.limit:
    pass
