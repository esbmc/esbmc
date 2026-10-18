# JSON carries b"ab" identically to "ab", so the runtime cannot tell them
# apart. Refused rather than treated as str.
x = b"ab"
assert len(x) == 2
