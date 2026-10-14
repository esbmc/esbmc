#include <cassert>
struct B;
struct R
{
  int a;
  const B *c;
};
struct B
{
  virtual R f(int v) const;
};
struct D : B
{
  R f(int v) const;
};
static D d;
R B::f(int v) const
{
  R r = {v, this};
  return r;
}
R D::f(int v) const
{
  R r = {v + 1, this};
  return r;
}
int main()
{
  const B *p = &d;
  R r = p->f(7);
  assert(r.a == 8 && r.c == p);
  return 0;
}
