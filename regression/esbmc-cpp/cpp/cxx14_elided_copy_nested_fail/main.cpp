// Before C++17, clang elides each copy in `C(C(x))`, not only the outermost.
#include <cassert>
int dtors = 0, copies = 0;
struct C
{
  int v;
  C(int x) : v(x)
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
};
int main()
{
  {
    C c = C(C(1));
  }
  assert(dtors == 2);
  return 0;
}
