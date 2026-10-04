// The copies clang elides before C++17 in each branch of a class conditional.
#include <cassert>
int ctors = 0, dtors = 0;
struct C
{
  int v;
  C(int x) : v(x)
  {
    ctors++;
  }
  C(const C &o) : v(o.v)
  {
    ctors++;
  }
  ~C()
  {
    dtors++;
  }
};
int nondet_int();
int main()
{
  int b = nondet_int();
  {
    C c = b ? C(1) : C(2);
    assert(c.v == (b ? 1 : 2));
  }
  assert(ctors == 1 && dtors == 1);
  {
    int w = (b ? C(3) : C(4)).v;
    assert(w == (b ? 3 : 4));
  }
  assert(ctors == 2 && dtors == 2);
  return 0;
}
