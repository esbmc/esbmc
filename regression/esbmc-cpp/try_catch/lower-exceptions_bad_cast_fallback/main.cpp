#include <cassert>

// A dynamic_cast<T&> site compiles to a call to the bodyless intrinsic
// __ESBMC_throw_bad_cast (the failure path). The lowering cannot see that
// hidden throw, so a function containing it must fall back to the imperative
// path rather than lower (and silently drop a bad_cast). Here the cast
// succeeds, so the program is well-defined and the verdict does not depend on
// the imperative bad_cast synthesis — keeping it environment-stable.
struct B
{
  int x;
  virtual ~B()
  {
  }
};
struct D : B
{
};

int main()
{
  D d;
  d.x = 7;
  B &b = d;
  try
  {
    D &r = dynamic_cast<D &>(b); // succeeds: dynamic type is D
    assert(r.x == 7);
  }
  catch (...)
  {
    assert(0); // not reached
  }
  return 0;
}
