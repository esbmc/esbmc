# Regression for #5658: int.from_bytes() on a symbolic `bytes` function
# parameter used to abort ESBMC entirely ("ERROR: __ESBMC_get_object_size:
# cannot determine the size of a non-array object") because the parameter
# decayed to a plain pointer with no backing object. The --function harness
# now allocates a nondet-length array for it, so this verifies soundly.
def g(data: bytes) -> int:
    return int.from_bytes(data, "little")
