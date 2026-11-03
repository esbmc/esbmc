// A braced aggregate of the variable's own class initialises the variable
// itself ([dcl.init]/17.6.1): no temporary is constructed or destroyed.
#include <cassert>
#include <cstdlib>

int dtors = 0;

struct A
{
  int *p;
  ~A()
  {
    ++dtors;
    free(p);
  }
};

struct B
{
  int k;
  A a;
};

int main()
{
  {
    A a = A{(int *)malloc(sizeof(int))};
    *a.p = 1;
    auto b = A{nullptr};
    B c = B{2, {(int *)malloc(sizeof(int))}};
    *c.a.p = 3;
    assert(dtors == 0);
  }
  assert(dtors == 3);
  return 0;
}
