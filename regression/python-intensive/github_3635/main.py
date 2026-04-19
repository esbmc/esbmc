def test_escape_and_unicode():
    s = "line1\nline2"
    assert s.splitlines() == ["line1", "line2"]

    u = "café"
    assert "é" in u
    assert u.encode("utf-8").decode("utf-8") == "café"

test_escape_and_unicode()
