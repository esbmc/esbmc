def func() -> int:
   return 1

x : int = func()
if (x == 0):
    y : int = 1/x
else:
    assert (x == 0)
