# str.format() with a bignum argument: format_value_from_json must trap
# rather than silently render the value as the string "None". Before the
# fix the null sentinel in the tagged Constant was indistinguishable from
# a real Python None on this path.
s = "x={}".format(18446744073709551616)
assert s == "x=18446744073709551616"
