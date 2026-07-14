# Pre-fix, as_integer_ratio() was undefined (assert(false) stub) and a
# tuple unpack of its result crashed conversion ("Cannot unpack empty").
# Now the literal fold renders the exact ratio, so this wrong claim must
# be a real FAILED.
assert (2.5).as_integer_ratio() == (5, 3)
