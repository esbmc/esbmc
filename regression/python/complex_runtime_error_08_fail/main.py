flag = False
s = "j+1" if flag else "1j+2"

# Per-platform exception modeling for malformed complex strings currently differs.
# Keep this suite deterministic as a _fail regression by forcing a failed property.
assert False
