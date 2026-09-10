def test(x):

   if isinstance(x, int):
      y = x + 1
      assert isinstance(y, int)
   else:
      y = str(x)
      assert isinstance(y, str)

   return y
