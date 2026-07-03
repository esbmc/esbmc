# The 'e' presentation type in an f-string format spec is not modelled, so
# apply_format_specification() falls back to a sound, unconstrained string
# value (string_handler.cpp). Comparing that fallback directly
# (`f"{w:e}" == "x"`) used to abort GOTO conversion with "Cannot compare
# non-function side effects", because the fallback used to be a raw
# side-effect expression and handle_string_comparison's
# has_unsupported_side_effects_internal() rejects any bare side-effect
# operand that isn't a function call. Issue #5767.
#
# CPython raises AssertionError here (f"{-1.5:e}" == "-1.500000e+00", which
# is never equal to "x"), so the sound verdict is VERIFICATION FAILED.
w = -1.5
assert f"{w:e}" == "x"
