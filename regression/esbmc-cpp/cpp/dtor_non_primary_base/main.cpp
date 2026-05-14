// Multiple inheritance: the non-primary base B2 sits at a non-zero offset
// inside D. ~D must call ~B2 with `this` adjusted to the B2 subobject so
// that B2 sees its own `y == 2`. build_destructor_chain casts the derived
// `this` to the base pointer type via gen_typecast, mirroring how the
// constructor base-init path delegates this elsewhere in the frontend.
// ESBMC's struct-layout model resolves the cast soundly, so this test
// pins the observable behaviour and guards against future regressions.
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
