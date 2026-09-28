#include <cassert>

int observed = 0;

struct Pad
{
  int p;
  Pad() : p(1) {}
};

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
  // 1 is Pad::p — the value ~Base() read when its `this` was left at the
  // start of the derived object. Pinning it keeps the wrong adjustment from
  // silently returning SUCCESSFUL again.
  assert(observed == 1);
  return 0;
}
