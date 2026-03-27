def test_dict_empty_iteration():
    d = {}
    counter = 0

    for k in d:
        counter += 1

    assert counter == 1


test_dict_empty_iteration()
