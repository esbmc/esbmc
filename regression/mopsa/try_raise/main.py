res = 1

try:
  raise TypeError
except TypeError:
  res = 2
