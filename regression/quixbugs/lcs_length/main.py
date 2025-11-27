
def lcs_length(s, t):
    from collections import Counter

    dp = Counter()

    for i in range(len(s)):
        for j in range(len(t)):
            if s[i] == t[j]:
                dp[i, j] = dp[i - 1, j - 1] + 1

    return max(dp.values()) if dp else 0

assert lcs_length("witch", "sandwich") == 2
assert lcs_length("meow", "homeowner") == 4
assert lcs_length("fun", "") == 0
assert lcs_length("fun", "function") == 3
assert lcs_length("cyborg", "cyber") == 3
assert lcs_length("physics", "physics") == 7
assert lcs_length("space age", "pace a") == 6
assert lcs_length("flippy", "floppy") == 3
assert lcs_length("acbdegcedbg", "begcfeubk") == 3