def test_negative_index_non_empty_list() -> None:
    lst: list[int] = [1]
    try:
        _ = lst[-1]
        assert False, "IndexError not raised"
    except IndexError:
        pass


test_negative_index_non_empty_list()
