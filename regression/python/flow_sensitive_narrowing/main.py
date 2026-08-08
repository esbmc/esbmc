def test(x):

   if isinstance(x, int):
      y = x + 1
      assert isinstance(y, int)
   else:
      y = str(x)
      assert isinstance(y, str)

   return y

test(5)
assert(test(5) == 6)
