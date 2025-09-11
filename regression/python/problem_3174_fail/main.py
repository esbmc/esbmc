def minChanges(s: str) -> int:
    # for each 2 char, if not equal one needs to be changed
    ans = i = 0
    while i < len(s):
        if s[i] != s[i + 1]:
            ans += 1
        i += 2
    return ans


name = "testing"
assert minChanges(name) == 1
