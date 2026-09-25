import numpy as np

# Known, tracked gap (not this test's own concern to fix): the second
# branch's np.ravel() correctly declines the pointer view (build_scalar_
# pointer_view refuses to overwrite the first branch's registration for
# the same target name b -- see view_branch_registration_conflict_success)
# and falls back to an independent copy, but numpy_pointer_view_info_ is a
# single frontend-only map with no per-branch state: len(b) still consults
# the first branch's leftover {length=3, stride=1} entry regardless of
# which branch actually ran. ADR-NP-003's own roadmap entry already notes
# this class of gap ("tracked by frontend-only maps ... not by
# ndarray_descriptor"); resolving it needs branch-aware or path-sensitive
# view metadata, out of scope for a "decline on re-registration" fix.


def pick(cond):
    a = np.array([1, 2, 3])
    c = np.array([10, 20, 30, 40, 50])
    if cond:
        b = np.ravel(a)
    else:
        b = np.ravel(c)
    return len(b)


assert pick(False) == 5
