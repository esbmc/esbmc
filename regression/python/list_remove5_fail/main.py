# Should raise ValueError (ESBMC: assert(0) fires)
l = [1, 2, 3]
l.remove(99)  # ESBMC should report a violation here
assert False  # unreachable
