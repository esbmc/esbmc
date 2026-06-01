# Subscripting the result of an inline list-returning call -- e.g.
# sorted(xs)[0] -- must not abort the frontend. The call result is a
# code_function_callt statement; indexing it built __ESBMC_list_size over the
# call statement, whose operand type is empty, aborting symex ("got empty,
# expected pointer"). The call is now materialised into a temporary first,
# matching the assigned form (s = sorted(xs); s[0]). See #4807 (humaneval/158).


def smallest(nums):
    return sorted(nums)[0]


assert smallest([3, 1, 2]) == 1
assert smallest([5, 4, 9, 2]) == 2

# Bound form (already worked) must still agree.
def smallest_bound(nums):
    s = sorted(nums)
    return s[0]


assert smallest_bound([3, 1, 2]) == 1
