# A module-level function whose return type is a user-defined class, called
# from a site that lexically precedes its definition (a forward reference).
# This is valid Python: `make` is resolved at call time, by which point it is
# defined. The frontend previously aborted with an assertion in
# handle_function_call_rhs because the callee symbol was not yet in the symbol
# table when the call site was converted.

class Widget:
    def __init__(self):
        self.x = 0.0


def f() -> float:
    w = make()
    w.x = 7.0
    return w.x


def make() -> Widget:
    return Widget()


assert f() == 7.0
