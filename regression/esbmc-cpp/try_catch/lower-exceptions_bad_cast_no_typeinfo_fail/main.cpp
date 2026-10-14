#include <cassert>

// Companion to lower-exceptions_bad_cast_no_typeinfo: the failing-cast path.
// Without <typeinfo> there is no std::bad_cast object to throw, so the lowering
// turns the __ESBMC_throw_bad_cast intrinsic into a cast-must-succeed assertion.
// Here the dynamic type is B, not D, so dynamic_cast<D&>(b) fails: a reference
// dynamic_cast with no RTTI model is std::terminate, reported as a verification
// failure where the cast occurs (rather than silently dropped). The assertion
// fires, so the program verifies FAILED.
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
  B b;
  b.x = 7;
  B &ref = b;
  D &r = dynamic_cast<D &>(ref); // fails: dynamic type is B, not D
  assert(r.x == 7);
  return 0;
}
