#include <cassert>

struct Base
{
  int bx;
};

struct Derived : Base
{
  int dy;
};

int main()
{
  // CK_BaseToDerivedMemberPointer: int Base::* -> int Derived::*
  int Base::*bp = &Base::bx;
  int Derived::*dp = static_cast<int Derived::*>(bp);

  Derived d;
  d.bx = 7;
  d.dy = 9;
  assert(d.*dp == 7);
  return 0;
}
