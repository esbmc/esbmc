#include <cassert>

// A dynamic_cast<T&> site compiles to a call to the bodyless intrinsic
// __ESBMC_throw_bad_cast (the failure path). The lowering rewrites that call
// into a std::bad_cast THROW only when <typeinfo>'s std::bad_cast is in the
// symbol table; this program never includes <typeinfo>, so the intrinsic is
// unlowerable. Lowering is the only exception path (#5075), so the program is
// reported as a hard error rather than miscompiled (silently dropping the
// bad_cast). Including <typeinfo> would let it lower (see _bad_cast_caught).
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
