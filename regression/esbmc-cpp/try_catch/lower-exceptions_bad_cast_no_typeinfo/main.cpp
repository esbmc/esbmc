#include <cassert>

// A dynamic_cast<T&> site compiles to a call to the bodyless intrinsic
// __ESBMC_throw_bad_cast on the failure path. The lowering rewrites that call
// into a std::bad_cast THROW when <typeinfo>'s std::bad_cast is in the symbol
// table; this program never includes <typeinfo>, so there is no exception
// object to throw and the call is lowered to a cast-must-succeed assertion
// instead. Here the cast succeeds (the dynamic type is D), so that assertion is
// unreachable and the program verifies SUCCESSFUL. The _fail companion takes the
// failing cast and shows the assertion firing.
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
  D &r = dynamic_cast<D &>(b); // succeeds: dynamic type is D
  assert(r.x == 7);
  return 0;
}
