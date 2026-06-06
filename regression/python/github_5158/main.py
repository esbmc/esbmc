# Joining a list comprehension whose elements come from iterating an
# unannotated string parameter (esbmc/esbmc#5158). The comprehension target
# is a 1-character string but was statically typed void*, so list.append()
# byte-copied (and corrupted) the element pointer instead of storing it as a
# string. The joined result must equal the filtered string.
def reverse_delete(s, c):
    s = ''.join([char for char in s if char not in c])
    return (s, s[::-1] == s)


assert reverse_delete("abcde", "ae") == ('bcd', False)
