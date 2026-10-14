# Companion to list_extend_str_len: pins the null-terminator subtraction.
# "hello" is 5 chars, so extending [0] yields 6 elements, not 7. A length of 7
# would only hold if the (char-array size - 1) subtraction wrongly counted the
# null terminator, so this assertion must fail.
def test():
    a = [0]
    a.extend("hello")
    assert len(a) == 7

test()
