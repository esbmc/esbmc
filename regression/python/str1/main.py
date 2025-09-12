def test_no_arguments():
    """Test str() with no arguments returns """""
    x = str()
    assert x == ""
    assert isinstance(x, str)

test_no_arguments()
