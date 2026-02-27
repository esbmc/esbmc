def test_indexing_and_slicing():
    l = [10, 20, 30, 40, 50]
    try:
        _ = l[0]
        assert False, "IndexError not raised"
    except IndexError:
        pass


test_indexing_and_slicing()
