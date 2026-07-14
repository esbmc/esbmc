#include <cassert>

// Symbolic exception lowering (#5075): a thrown object is caught by reference
// and its value flows through __ESBMC_exc_value into the handler variable.
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
  catch (A &a)
  {
    assert(a.x == 42);
    return 1;
  }
  return 0;
}
