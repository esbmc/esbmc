# Negative variant: math.sqrt() of a negative "any"-typed value raises
# ValueError (math domain error). The domain-error path must stay reachable
# (it is guarded by a nondet condition for any-typed operands), so the
# uncaught exception is reported instead of being silently dropped -- and the
# frontend must not abort.

import math


def use(p):
    return math.sqrt(p)


def g():
    f = [-4, 9]
    f.append(16)
    return use(f[0])


g()
