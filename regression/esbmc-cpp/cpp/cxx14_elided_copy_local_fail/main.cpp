// The copies clang elides before C++17, in the return and the initialisation.
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
    C c = C::make(3);
  }
  assert(dtors == 2);
  return 0;
}
