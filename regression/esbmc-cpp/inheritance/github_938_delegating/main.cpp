// esbmc/esbmc#938, delegating-constructor edge: a delegating constructor must
// forward its own completeness to the target constructor (C1->C1, C2->C2).
// Here B() delegates to B(7); when B is a base subobject of D, B() is a
// base-object constructor, so the delegated B(7) must NOT re-initialise the
// virtual base A. D (most-derived) default-constructs A, leaving A::m == 0.
#include <cassert>

struct A
{
  int m;
  A() : m(0) {}
  A(int x) : m(x) {}
};

struct B : virtual A
{
  B(int x) : A(x) {}
  B() : B(7) {}
};

struct D : B
{
  D() : B() {}
};

int main()
{
  D d;
  assert(d.m == 0);
  return 0;
}
