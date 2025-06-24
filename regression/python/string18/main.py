def identity_string(x: str) -> str:
    return x

def test_unicode():
    unicode_str = "Hello 世界 🌍"
    assert identity_string(unicode_str) == unicode_str

def test_very_long_string():
    very_long = "x" * 1000  # 1000 character string
<<<<<<< HEAD
<<<<<<< HEAD
=======
    assert identity_string(very_long) == very_long
>>>>>>> 2b11dd71d ([python] enhanced string comparison handling (#2497))
=======
>>>>>>> 928d8df54 ([python] avoid string comparison with sideeffects)

def complex_identity(x: str) -> str:
    temp = x
    if True:
        result = temp
    return result

def test_complex_identity():
    test_str = "complex"
    assert complex_identity(test_str) == test_str

def multi_return_identity(x: str, flag: bool) -> str:
    if flag:
        return x
    else:
        return x

def test_multi_return():
    test_str = "multi_return"
<<<<<<< HEAD
<<<<<<< HEAD
=======
    assert multi_return_identity(test_str, True) == test_str
    assert multi_return_identity(test_str, False) == test_str
>>>>>>> 2b11dd71d ([python] enhanced string comparison handling (#2497))
=======
>>>>>>> 928d8df54 ([python] avoid string comparison with sideeffects)

def fake_identity(x: str) -> str:
    if len(x) > 0:
        return x
    else:
        return "default"

def test_fake_identity():
    test_str = "fake"
    result = fake_identity(test_str)

def recursive_identity(x: str, depth: int = 0) -> str:
    if depth <= 0:
        return x
    return recursive_identity(x, depth - 1)

def test_recursive():
    test_str = "recursive"
<<<<<<< HEAD
<<<<<<< HEAD
=======
    assert recursive_identity(test_str, 0) == test_str
>>>>>>> 2b11dd71d ([python] enhanced string comparison handling (#2497))
=======
>>>>>>> 928d8df54 ([python] avoid string comparison with sideeffects)

def test_null_bytes():
    null_str = "test\0null"
    assert identity_string(null_str) == null_str

def test_deep_nesting():
    nested_str = "deep"
    result = identity_string(identity_string(identity_string(identity_string(nested_str))))
    assert result == nested_str

def annotated_identity(s: str) -> str:
    return s

def test_annotations():
    ann_str = "annotated"
    assert annotated_identity(ann_str) == ann_str
