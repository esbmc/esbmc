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
};
int take(C c)
{
  return c.v;
}
int main()
{
  int r = take(C::make(3));
  assert(r == 3);
  assert(dtors == 2);
  return 0;
}
