# Issue #8347: abs() and round() over a call on an importable module.
import helper

assert abs(helper.f()) == 3
assert round(helper.g()) == -3
