# KNOWNBUG: calling a method that reads self.<attr> on an inline (unnamed)
# class instance returns a wrong result. The named form (c = C(5); c.get())
# works, and inline attribute access (C(5).x, PR #5821) works, but an inline
# instance is not materialised as the method's `self`, so self.x reads
# garbage. A method whose name collides with a dict method (get/pop/...)
# additionally misroutes to the dict handler ("Dictionary variable not
# found"). The fix needs the method-dispatch path (function_call/expr.cpp,
# obj_symbol resolution) to spill an inline receiver into a temp used as self.
class C:
    def __init__(self, v):
        self.x = v

    def compute(self):
        return self.x * 2


assert C(5).compute() == 10
