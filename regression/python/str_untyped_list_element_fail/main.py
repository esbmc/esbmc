# Negative variant of str_untyped_list_element: the same str()/arithmetic over
# an unresolved list element must still be modelled soundly. The empty list sums
# to 0, so asserting it equals 1 is a genuine violation ESBMC must catch rather
# than mask behind the previous frontend abort. See esbmc/esbmc#4807.
def f(lst):
    return sum([i**2 for i in lst if i > 0 and i % 2 != 0 and "." not in str(i)])


assert f([]) == 1
