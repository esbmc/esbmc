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
  }
  assert(copies == 1);
  return 0;
}
