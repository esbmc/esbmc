// Pointer-form displacement onto a non-primary base of a heap object. Also
// pins the cpp_new round trip: back_sideeffect used to write an empty "#size",
// which the next migration read in preference to the real one.
#include <cassert>

struct B1
{
  virtual ~B1()
  {
  }
  int x;
  B1() : x(1)
  {
  }
};

struct B2
{
  virtual ~B2()
  {
  }
  int y;
  B2() : y(2)
  {
  }
};

struct D : B1, B2
{
};

int main()
{
  D *d = new D();
  B2 *b2 = d;
  assert(b2->y == 1);
  delete d;
}
