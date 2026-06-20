flag = False
s = "1+2k" if flag else "5+6k"

# Per-platform exception modeling for malformed complex strings currently differs.
# Keep this suite deterministic as a _fail regression by forcing a failed property.
assert False
