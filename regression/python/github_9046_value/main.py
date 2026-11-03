# A module-qualified function value names that module's function (#9046)
import one
import two


def call(fn) -> int:
    return fn()


assert call(one.f) == 1
assert call(two.f) == 2
