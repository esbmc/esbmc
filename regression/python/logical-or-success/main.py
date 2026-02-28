def fail() -> None:
    assert (False)


x: int = 1
y: int = 2
z: int = 3

if (x == 0 or y == 1 and z == 2):
    fail()
