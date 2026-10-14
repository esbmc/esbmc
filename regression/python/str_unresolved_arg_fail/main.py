# Negative variant of str_unresolved_arg: an empty input never calls the
# helper, so the count is 0; asserting it equals 1 is a genuine violation
# ESBMC must catch rather than mask behind the previous frontend abort.
# See esbmc/esbmc#4807.
def count_void(arr):

    def digits(n):
        return [int(i) for i in str(n)]

    return len([digits(i) for i in arr])


assert count_void([]) == 1
