# Operations over an element of a list whose element type cannot be statically
# resolved. Calling these helpers with [] leaves the comprehension loop variable
# typed as the generic list pointer (the documented list[int] default), so the
# arithmetic (i ** 2, i > 0, i % 2) and str(i) in the comprehensions below must
# not abort GOTO generation even though the loop bodies are dead for the empty
# list. See esbmc/esbmc#4807 (humaneval_151).
def double_the_difference(lst):
    return sum([i**2 for i in lst if i > 0 and i % 2 != 0 and "." not in str(i)])


def stringify(lst):
    return [str(x) for x in lst]


# Empty lists: the loop bodies are dead, so the results are the empty sum / list.
assert double_the_difference([]) == 0
assert stringify([]) == []
