#include <cassert>

// The bound handler variable carries the real thrown value (not nondet), so an
// assertion contradicting it must fail under symbolic exception lowering.
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
    assert(a.x == 43);
    return 1;
  }
  return 0;
}
