# Negative companion to conditional_retype_inblock: an in-block read of a
# variable retyped from int to str must still be checked against the real
# string value, so a wrong in-block assertion is detected (the retype must not
# silently mask genuine violations).
a = 1
if True:
    a = "Rafael"
    assert a == "WRONG"
