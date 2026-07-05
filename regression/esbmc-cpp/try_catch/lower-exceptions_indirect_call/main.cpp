#include <cassert>

// An indirect (function-pointer) call whose target throws must still propagate:
// the call site is guarded even though the callee is not statically known.
struct E
{
  int v;
  E(int a) : v(a)
  {
  }
};

void thrower()
{
  throw E(9);
}

int main()
{
  void (*fp)() = thrower;
  try
  {
    fp();
  }
  catch (E &e)
  {
    assert(e.v == 9);
    return 1;
  }
  assert(0); // unreachable unless the exception silently vanished
  return 0;
}
