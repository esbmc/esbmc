def starred_unpack():
    head, *tail = [1, 2, 3]
    assert head == 1


starred_unpack()
