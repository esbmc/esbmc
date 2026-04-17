class C:
  def __init__(self, x):
    self.x = x

  def f(self):
    return self.x

class D:
  x = 100
  def g(self):
    return self.x

c = C(1)
f1 = c.f
c = C(2)
c.x = 40
f2 = c.f
d = D()
res = f1() + f2() + d.g()
