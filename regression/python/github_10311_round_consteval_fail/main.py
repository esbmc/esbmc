# #10311: round() folded inside an assert ties to even only on an exact half.
k = nondet_int()
if k == 0:
    assert round(2.5000000000001) == 2
elif k == 1:
    assert round(-2.5000000000001) == -2
elif k == 2:
    assert round(1.95, 1) == 2.0
elif k == 3:
    assert round(-0.445, 2) == -0.44
elif k == 4:
    assert round(9007199254740993, 0) == 9007199254740992
elif k == 5:
    assert round(123456789012345666, -1) == 123456789012345664
else:
    y = round(123456789012345666, -1)
    assert y == 123456789012345664
