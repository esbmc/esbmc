# Regression for #4807: appending an empty string literal to a list used to
# crash GOTO conversion with `migrate expr failed` on a malformed
# constant{type=char*, op0=char '\0'} produced by the `char[1] -> char*`
# in-place type rewrite in handle_list_append. The same shape was reached
# whenever `''.join(<empty>)` returned the empty char[1] string.
#
# This test only checks that conversion completes and the list has the
# expected length; element-equality semantics for empty / join-produced
# strings are tracked separately and are not the point of this fix.

def f():
    result = []
    result.append("")
    return result


def g():
    current = []
    out = []
    out.append(''.join(current))  # ''.join(<empty>) used to crash on append
    return out


if __name__ == "__main__":
    a = f()
    assert len(a) == 1

    b = g()
    assert len(b) == 1
