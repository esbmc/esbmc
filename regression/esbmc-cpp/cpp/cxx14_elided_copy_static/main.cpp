// An elided copy initialising a global or a static local.
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
C g = C::make(8);
int f()
{
  static C s = C::make(7);
  return s.v;
}
int main()
{
  assert(g.v == 8 && f() == 7 && copies == 0);
  return 0;
}
