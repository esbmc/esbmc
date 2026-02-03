def gen():
  i = 1
  return
  yield 1
  return

g = gen()
s = 1
try:
  s = s + next(g)
except:
  s = 5
res = s


