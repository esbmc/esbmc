s = input()
__ESBMC_assume("," not in s)
parts = (s + ",end").split(",", 1)
assert len(parts) == 2
assert parts[0] == s
assert parts[1] == "end"
