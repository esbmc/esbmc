#include <cassert>

int copies = 0;

struct C
{
  int v;
  C(int x) : v(x)
  {
  }
  C(const C &o) : v(o.v + 100)
  {
    copies++;
  }
};

struct A
{
  int a = 1;
};

struct B
{
  int b;
  B(int x) : b(x)
  {
  }
  B(const B &o) : b(o.b + 100)
  {
  }
};

struct D : A, B
{
  D() : B(7)
  {
  }
};

struct E
{
  int w, v;
  E(int x) : w(x), v(0)
  {
  }
  E(const E &o) : w(0), v(o.w)
  {
  }
};

int main()
{
  try
  {
    throw C(1);
  }
  catch (C c)
  {
    assert(c.v == 101);
  }
  assert(copies == 1);

  try
  {
    throw D();
  }
  catch (const B b)
  {
    assert(b.b == 107);
  }

  try
  {
    throw E(5);
  }
  catch (E e)
  {
    assert(e.w == 0 && e.v == 5);
  }
  return 0;
}
