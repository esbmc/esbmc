// The copy of a by-value argument that clang elides before C++17.
#include <cassert>
int dtors = 0, copies = 0;
struct C
{
  int v;
  C(int x = 0) : v(x)
  {
  }
  C(const C &o) : v(o.v)
  {
    copies++;
  }
  ~C()
  {
    dtors++;
  }
  static C make(int x)
  {
    return C(x);
  }
  int add(C o)
  {
    return v + o.v;
  }
  int operator*(C o)
  {
    return v * o.v;
  }
};
struct D
{
  int w;
  D(C c) : w(c.v)
  {
  }
};
int take(C c)
{
  return c.v;
}
int main()
{
  int r = take(C::make(3));
  assert(r == 3);
  assert(dtors == 1 && copies == 0);
  C a(2);
  int s = a.add(C::make(2));
  assert(s == 4);
  assert(dtors == 2 && copies == 0);
  int t = a * C::make(5);
  assert(t == 10);
  assert(dtors == 3 && copies == 0);
  int w = D(C::make(6)).w;
  assert(w == 6);
  assert(dtors == 4 && copies == 0);
  return 0;
}
