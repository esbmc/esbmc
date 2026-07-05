#include <cassert>

// Throw in a callee, caught in the caller: the exception state propagates out
// of f through the return and f's may-throw call site is re-checked in main.
struct X
{
  int v;
  X(int a) : v(a)
  {
  }
};

void f()
{
  throw X(7);
}

int main()
{
  try
  {
    f();
  }
  catch (X &x)
  {
    assert(x.v == 7);
    return 1;
  }
  return 0;
}
