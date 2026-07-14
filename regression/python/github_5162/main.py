# A List[List[float]] parameter annotation used to type the inner element as a
# bogus "tag-List" struct (typing.List was not mapped to the PyListObject type
# the lowercase builtin "list" resolves to). len(A[0]) then misrouted to
# strlen(), aborting BMC with "argument ... strlen ... type mismatch: got
# struct, expected pointer" (#5162). The element type now resolves correctly,
# so len() of a nested-list subscript routes to __ESBMC_list_size.
from typing import List


def k(A: List[List[float]]) -> int:
    w = len(A[0])
    out = [0.0 for _ in range(w)]
    return len(out)


A = [[1.0], [2.0]]
assert k(A) == 1


# len() of each inner list of differing lengths is computed independently.
def widths(M: List[List[float]]) -> int:
    return len(M[0]) + len(M[1])


B = [[1.0, 2.0, 3.0], [4.0]]
assert widths(B) == 4


# The capitalised typing alias Set shares the root cause (it was typed as a
# bogus "tag-Set" struct inside a list); len() of a nested set must work too.
from typing import Set


def sizes(S: List[Set[int]]) -> int:
    return len(S[0])


C = [{1, 2}]
assert sizes(C) == 2
