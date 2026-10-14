# Extending a list with a string appends one element per character. The
# character count is computed as (char-array size - 1), dropping the null
# terminator -- the subtraction migrated to IREP2 in build_extend_list_call.
# "hello" is 5 chars, so the result has 1 + 5 = 6 elements.
def test():
    a = [0]
    a.extend("hello")
    assert len(a) == 6

test()
