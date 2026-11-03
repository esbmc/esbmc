// A braced aggregate of the variable's own class initialises the variable
// itself ([dcl.init]/17.6.1), so no destructor has run yet: the assertion
// fails natively.
#include <cassert>

int dtors = 0;

struct A
{
  int k;
  ~A()
  {
    ++dtors;
  }
};

int main()
{
  {
    A a = A{1};
    assert(dtors == 1);
  }
  return 0;
}
