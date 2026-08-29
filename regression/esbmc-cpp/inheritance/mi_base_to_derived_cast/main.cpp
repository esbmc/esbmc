#include <cassert>

struct Pad
{
  int p;
  Pad() : p(1) {}
};

struct Base
{
  int q;
  Base() : q(5) {}
  virtual ~Base() {}
};

// Base is not the first base, so &d and (Base *)&d differ. Casting back must
// undo exactly that displacement, otherwise the derived fields are read from
// the wrong slot (#1866).
struct Der : Pad, Base
{
  int y;
  Der() : y(42) {}
};

int main()
{
  Der d;

  Base *b = &d;
  assert(b->q == 5);

  Der *dd = static_cast<Der *>(b);
  assert(dd->y == 42);
  assert(dd->p == 1);
  assert(dd->q == 5);
  assert(dd == &d);

  // Reference form of the same round trip.
  Base &rb = d;
  Der &rd = static_cast<Der &>(rb);
  assert(rd.y == 42);

  return 0;
}
