# Regression: width/fill string methods (center/ljust/rjust/zfill) and
# expandtabs route their integer argument through get_constant_int(), whose
# guard was inverted (`!to_integer` — to_integer returns false on success).
# Every valid constant width was rejected, so these methods silently fell back
# to a nondet/default result and proved false assertions on their length/value.
assert "42".zfill(5) == "00042"
assert "hi".center(6, "*") == "**hi**"
assert "hi".ljust(5, ".") == "hi..."
assert "hi".rjust(5, ".") == "...hi"
assert "a\tb".expandtabs(4) == "a   b"
assert "x".center(7) == "   x   "
