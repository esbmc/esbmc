# No final else, so x/y should stay plain scalars, not get tagged. x + y
# would fail to convert if both got tagged (Add between two tagged
# operands isn't supported), so this pins the untagged case rather than
# checking a value that would hold either way.
cond1 = nondet_bool()
cond2 = nondet_bool()
if cond1:
    x = 1
    y = 10
    z = x + y
    assert z == 11
elif cond2:
    x = "a"
    y = "b"
    z = x + y
    assert z == "ab"
