#include <cassert>

// Nested try: the thrown B does not match the inner catch(A&) and must
// propagate to the outer catch(B&). Symbolic lowering matches a throw against
// its full enclosing-try chain (innermost outward), so this is caught and the
// value flows through the outer handler.
struct A
{
  int x;
  A(int v) : x(v)
  {
  }
};
struct B
{
  int y;
  B(int v) : y(v)
  {
  }
};

int main()
{
  try
  {
    try
    {
      throw B(2);
    }
    catch (A &a)
    {
      return 1;
    }
  }
  catch (B &b)
  {
    assert(b.y == 2);
    return 2;
  }
  return 0;
}
