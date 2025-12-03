
def longest_common_subsequence(a, b):
    if not a or not b:
        return ''

    elif a[0] == b[0]:
        return a[0] + longest_common_subsequence(a[1:], b[1:])

    else:
        return max(
            longest_common_subsequence(a, b[1:]),
            longest_common_subsequence(a[1:], b),
            key=len
        )

assert longest_common_subsequence("headache", "pentadactyl") == "eadac"
assert longest_common_subsequence("daenarys", "targaryen") == "aary"
assert longest_common_subsequence("XMJYAUZ", "MZJAWXU") == "MJAU"
assert longest_common_subsequence("thisisatest", "testing123testing") == "tsitest"
assert longest_common_subsequence("1234", "1224533324") == "1234"
assert longest_common_subsequence("abcbdab", "bdcaba") == "bcba"
assert longest_common_subsequence("TATAGC", "TAGCAG") == "TAAG"
assert longest_common_subsequence("ABCBDAB", "BDCABA") == "BCBA"
assert longest_common_subsequence("ABCD", "XBCYDQ") == "BCD"
assert longest_common_subsequence("acbdegcedbg", "begcfeubk") == "begceb"