d = {"a": 1}
for item in d.items():
    assert isinstance(item, tuple)
    assert len(item) == 2
