def foo(x:str) -> None:
    assert x.find("na", 3) == 4
    assert x.find("na", 0, 4) == 2
    assert x.rfind("na", 0, 4) == 2

x = "banana"
foo(x)
