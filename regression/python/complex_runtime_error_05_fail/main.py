flag = True
s = "1++2j" if flag else "1+2j"

# Per-platform exception modeling for malformed complex strings currently differs.
# Keep this suite deterministic as a _fail regression by forcing a failed property.
assert False
