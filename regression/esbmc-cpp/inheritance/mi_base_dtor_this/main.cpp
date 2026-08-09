#include <cassert>

int observed = 0;

struct Pad
{
  int p;
  Pad() : p(1) {}
};

// Clang's ABI promotes the polymorphic base to primary and lays it out at
// offset 0, while ESBMC nests base subobjects in declaration order. A base
// destructor `this` derived from getBaseClassOffset therefore lands on Pad's
// storage; it must come from the @base@ subobject address instead (#1866).
struct Base
{
  int q;
  Base() : q(5) {}
  virtual ~Base()
  {
    observed = q;
  }
};

struct Der : Pad, Base
{
  int y;
  Der() : y(42) {}
};

int main()
{
  {
    Der d;
  }
  assert(observed == 5);
  return 0;
}
