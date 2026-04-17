def test_case_operations():
    s = "PyThOn"
    assert s.lower() == "python"
    assert s.upper() == "PYTHON"
    assert s.swapcase() == "pYtHoN"
    assert "python".capitalize() == "Python"


test_case_operations()
