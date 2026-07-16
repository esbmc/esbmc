// Multiple inheritance: the non-primary base B2 sits at a non-zero offset
// inside D. ~D must call ~B2 with `this` adjusted to the B2 subobject so
// that B2 sees its own `y == 2`. The pointer cast build_destructor_chain
// emits does not apply the base-subobject offset (github #6021), so ~B2
// reads B1's storage. This was masked while destructors were skipped on
// explicit return (github #6077); with that fixed the assert executes and
// exposes the missing this-adjustment. KNOWNBUG until #6021 is resolved.
#include <cassert>

struct B1
{
  int x;
};

struct B2
{
  int y;
  ~B2() { assert(y == 2); }
};

struct D : B1, B2
{
  D()
  {
    x = 1;
    y = 2;
  }
};

int main()
{
  D d;
  return 0;
}
