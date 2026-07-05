#include <cassert>

// Catch by value: the handler variable is copy-constructed from the thrown
// object (v = *(T*)__ESBMC_exc_value), not bound by reference.
struct A
{
  int x;
  A(int v) : x(v)
  {
  }
};

int main()
{
  try
  {
    throw A(42);
  }
  catch (A a)
  {
    assert(a.x == 42);
    return 1;
  }
  return 0;
}
