// #8679: a call through a pointer to a virtual member function dispatches on
// the object's dynamic type.
#include <cassert>

struct B
{
  virtual int g()
  {
    return 1;
  }
};

struct D : B
{
  int g() override
  {
    return 2;
  }
};

int main()
{
  D d;
  B *pb = &d;
  int (B::*pg)() = &B::g;
  assert((pb->*pg)() == 1);
  return 0;
}
