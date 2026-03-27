def test_replace_and_find():
    s = "banana"
    assert s.replace("a", "o") == "bonono"
    assert s.count("na") == 2


test_replace_and_find()
