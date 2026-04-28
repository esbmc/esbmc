s = input()
__ESBMC_assume("," not in s)
parts = (s + ",end").split(",", maxsplit=1)
assert parts[0] == s
assert parts[1] == "end"
