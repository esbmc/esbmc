# Regression for #4807: `x = list()` (zero-argument constructor) used to
# crash GOTO conversion with `migrate expr failed` on a malformed
# constant{type=PyListObj*} produced by the generic call builder, because
# converter_funcall's `list(...)` dispatcher required args.size() == 1.
# The fix lowers `list()` to the same empty `[]` literal path used elsewhere.


def f():
    return list()


def g():
    lis = list()
    lis.append(7)
    return lis


if __name__ == "__main__":
    assert f() == []
    assert g() == [7]
