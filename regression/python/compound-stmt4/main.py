def test_all_compound_ops():
    a = 4
    a += 1  # 5
    a -= 2  # 3
    a *= 3  # 9
    a //= 2  # 4
    a %= 3  # 1

    b = 6
    b &= 3  # 2
    b |= 4  # 6
    b ^= 2  # 4
    b <<= 1  # 8
    b >>= 2  # 2

    assert a == 1
    assert b == 2


test_all_compound_ops()
