def longest_common_subsequence(a, b):
    if not a or not b:
        return ''

    elif a[0] == b[0]:
        return a[0] + longest_common_subsequence(a[1:], b)

    else:
        return max(
            longest_common_subsequence(a, b[1:]),
            longest_common_subsequence(a[1:], b),
            key=len
        )



"""
Longest Common Subsequence


Calculates the longest subsequence common to the two input strings. (A subsequence is any sequence of letters in the same order
they appear in the string, possibly skipping letters in between.)

Input:
    a: The first string to consider.
    b: The second string to consider.

Output:
    The longest string which is a subsequence of both strings. (If multiple subsequences of equal length exist, either is OK.)

Example:
    >>> longest_common_subsequence('headache', 'pentadactyl')
    'eadac'
"""

assert longest_common_subsequence('headache', 'pentadactyl') == 'eadac'
assert longest_common_subsequence('daenarys', 'targaryen') == 'aary'
assert longest_common_subsequence('XMJYAUZ', 'MZJAWXU') == 'MJAU'
assert longest_common_subsequence('thisisatest', 'testing123testing') == 'tsitest'
assert longest_common_subsequence('1234', '1224533324') == '1234'
assert longest_common_subsequence('abcbdab', 'bdcaba') == 'bcba'
assert longest_common_subsequence('TATAGC', 'TAGCAG') == 'TAAG'
assert longest_common_subsequence('ABCBDAB', 'BDCABA') == 'BCBA'
assert longest_common_subsequence('ABCD', 'XBCYDQ') == 'BCD'
assert longest_common_subsequence('acbdegcedbg', 'begcfeubk') == 'begceb'
