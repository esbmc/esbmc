// A braced mem-initializer holding a call:
// one object, one destructor ([dcl.init]/17.6.1).
#include <cassert>
int dtors = 0;
struct C
{
  int *p;
  C(int *q = nullptr) : p(q)
  {
  }
  ~C()
  {
    dtors++;
  }
  static C make(int *q)
  {
    return C{q};
  }
};
int x;
struct M
{
  C impl_;
  M() : impl_{C::make(&x)}
  {
  }
};
int main()
{
  {
    M m;
    assert(m.impl_.p == &x);
  }
  assert(dtors == 1);
  return 0;
}
