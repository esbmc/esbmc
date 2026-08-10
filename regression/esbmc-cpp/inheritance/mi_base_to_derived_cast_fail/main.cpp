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

struct Der : Pad, Base
{
  int y;
  Der() : y(42) {}
};

int main()
{
  Der d;
  Base *b = &d;

  // The downcast must land back on `d`, so this write is observable through
  // `d`. An unadjusted cast writes past the object instead and leaves y at
  // 42 — pinning the stale value keeps that from returning SUCCESSFUL again.
  Der *dd = static_cast<Der *>(b);
  dd->y = 7;
  assert(d.y == 42);

  return 0;
}
