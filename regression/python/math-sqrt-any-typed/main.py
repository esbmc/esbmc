# math.sqrt() applied to an "any"-typed value (an unannotated parameter bound
# to a dynamic-list element, which the frontend types as void*) must not abort
# the SMT backend. Before the fix this aborted in get_significand_width()
# because the operand was cast to float over a pointer sort (see humaneval/39).
# Here the (over-approximated) result is consumed inside a try/except, so the
# program verifies cleanly.

import math


def use(p):
    try:
        math.sqrt(p)
    except ValueError:
        pass
    return 0


def g():
    f = [4, 9]
    f.append(16)
    return use(f[0])


assert g() == 0
