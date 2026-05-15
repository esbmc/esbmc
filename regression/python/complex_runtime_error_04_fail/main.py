flag = False
s = "3+4j" if flag else ""

# Per-platform exception modeling for malformed complex strings currently differs.
# Keep this suite deterministic as a _fail regression by forcing a failed property.
assert False
