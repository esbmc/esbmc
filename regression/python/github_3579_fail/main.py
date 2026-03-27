def test_split_and_join():
    s = "a,b,c"
    parts = s.split(",")
    assert ".".join(parts) == s


test_split_and_join()
