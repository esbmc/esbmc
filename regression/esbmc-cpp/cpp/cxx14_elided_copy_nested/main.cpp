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
C make(int x)
{
  return C(C(x));
}
int main()
{
  {
    C c = C(C(C(1)));
    assert(c.v == 1);
  }
  assert(copies == 0 && dtors == 1);
  {
    C d = make(2);
    assert(d.v == 2);
  }
  assert(copies == 0 && dtors == 2);
  C a(3);
  C b = C(a);
  assert(b.v == 3 && copies == 1);
  return 0;
}
