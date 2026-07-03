x = 7
# Pre-fix, both the consteval JoinedStr fold and the string handler
# dropped the format spec, folding f"{x:03d}" to "7" and verifying this
# false assertion as SUCCESSFUL. CPython renders "007", so this must be
# a real FAILED.
assert f"{x:03d}" == "7"
