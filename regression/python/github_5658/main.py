# Regression for #5658: int.from_bytes() on a symbolic `bytes` function
# parameter used to abort ESBMC entirely ("ERROR: __ESBMC_get_object_size:
# cannot determine the size of a non-array object") because the parameter
# decays to a plain pointer with no compile-time array bound, and len()
# routed through __ESBMC_get_object_size expected an array. It must now
# reach a verdict instead of aborting -- memory safety of indexing an
# unconstrained bytes pointer is a separate, harder guarantee, so the
# expected verdict here is a genuine dereference failure, not a crash.
def g(data: bytes) -> int:
    return int.from_bytes(data, "little")
