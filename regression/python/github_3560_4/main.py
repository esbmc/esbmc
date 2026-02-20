s = input()
__ESBMC_assume("|" not in s)
parts = (s + "|tail").split("|", 1)
assert parts[0] == s
assert parts[1] == "tail"
