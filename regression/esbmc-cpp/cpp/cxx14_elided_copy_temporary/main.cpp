// A local initialised from a temporary of its own class.
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
int main()
{
  {
    C c = C(5);
    assert(c.v == 5);
  }
  assert(dtors == 1 && copies == 0);
  return 0;
}
