def test_negative_index_empty_list() -> None:
    lst: list[int] = []
    try:
        _ = lst[-1]
        assert False, "IndexError not raised"
    except IndexError:
        pass


test_negative_index_empty_list()
