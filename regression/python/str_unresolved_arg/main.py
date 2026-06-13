# str() of an argument whose type the frontend cannot statically resolve — an
# unannotated helper parameter reached only on a dead path, left as either the
# any_type (void*) or, after dynamic retyping, the generic list pointer — must
# be handled gracefully (treated as an int) instead of aborting conversion with
# "TypeError: str() expects a string argument". See esbmc/esbmc#4807 (108).
def count_void(arr):

    def digits(n):
        return [int(i) for i in str(n)]

    return len([digits(i) for i in arr])


def count_listptr(arr):

    def digits_sum(n):
        neg = 1
        if n < 0:
            n, neg = -1 * n, -1
        n = [int(i) for i in str(n)]
        n[0] = n[0] * neg
        return sum(n)

    return len([digits_sum(i) for i in arr])


# Empty input: the helpers are never called, so both reductions are 0 and the
# dead str() bodies must still convert.
assert count_void([]) == 0
assert count_listptr([]) == 0
