#include <cassert>

struct A
{
  int a;
};

struct B
{
  int b;
};

int main()
{
  // CK_ReinterpretMemberPointer round-trip: int A::* -> int B::* -> int A::*.
  int A::*pa = &A::a;
  int B::*pb = reinterpret_cast<int B::*>(pa);
  int A::*pa2 = reinterpret_cast<int A::*>(pb);

  // The frontend used to reject the cast outright; this asserts that the
  // round-tripped pointer still resolves to A::a when dereferenced.
  A x;
  x.a = 13;
  assert(x.*pa2 == 13);
  return 0;
}
