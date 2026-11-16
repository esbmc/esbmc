# #10311: round(x, n) raises OverflowError when the result is too large.
try:
    a = round(1.7976931348623157e308, -308)
    assert False
except OverflowError:
    pass
