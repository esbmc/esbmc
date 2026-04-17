# Bool global declared after class; method returns it and caller branches.
# Verifies that both True and False paths through the method are reachable.

class Guard:
    def is_enabled(self) -> bool:
        return enabled


enabled: bool = nondet_bool()

g = Guard()
if g.is_enabled():
    assert enabled
else:
    assert not enabled
