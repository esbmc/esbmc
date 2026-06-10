#include <cassert>

// `throw;` re-raises the in-flight exception from a handler; it propagates out
// of f and is caught in main with the original value intact.
struct X
{
  int v;
  X(int a) : v(a)
  {
  }
};

void f()
{
  try
  {
    throw X(1);
  }
  catch (X &x)
  {
    throw;
  }
}

int main()
{
  try
  {
    f();
  }
  catch (X &x)
  {
    assert(x.v == 1);
    return 1;
  }
  return 0;
}
