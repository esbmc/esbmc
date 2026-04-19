def foo() -> None:
    assert (None == None)
    assert not (None != None)
    assert (None is None)
    assert not (None is not None)

foo()
