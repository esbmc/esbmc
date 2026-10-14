# Negative companion to github_5158: the joined result is now correctly
# 'bcd', so asserting a wrong expected value must make verification FAIL.
# Before the fix this assertion spuriously "held" because the element was
# corrupted garbage that compared unequal to everything.
def reverse_delete(s, c):
    s = ''.join([char for char in s if char not in c])
    return (s, s[::-1] == s)


assert reverse_delete("abcde", "ae") == ('XYZ', False)
