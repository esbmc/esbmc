def f(x):
  global i
  i = x
  return i
i = 1
j = i + f(3) + i
