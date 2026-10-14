// KNOWNBUG: a virtual base whose own base has state is constructed at the
// wrong address, so nothing in that subobject is initialised.
//
// The GOTO for Mid's constructor shows the cause:
//
//     FUNCTION_CALL:  Base((struct Base *)this)
//
// `this` (a Mid*) is cast straight to Base* with no virtual-base offset
// adjustment. Under virtual inheritance the Base subobject is not at offset 0,
// so Base's constructor -- and transitively Root's -- writes to the wrong
// place and both members read back as unconstrained values.
//
// Narrowed from the stream models, where it makes a freshly constructed
// std::ostream's ios state indeterminate (ostream : virtual public ios,
// ios : public ios_base): std::ios on its own is fine, std::ostream is not.
// See regression/esbmc-cpp/cpp/ios_default_state_knownbug.
//
// The trigger is precisely "virtual base that itself has a base":
//   * with no grandparent (Base has no base of its own) it works;
//   * with the inheritance made non-virtual it works;
//   * virtual destructors are irrelevant -- it fails with and without them.
//
// Verified against clang++ -std=c++17 -fsanitize=address,undefined: exits 0.
#include <cassert>

struct Root
{
  int r;
  Root() : r(1)
  {
  }
};

struct Base : public Root
{
  int v;
  Base() : v(7)
  {
  }
};

struct Mid : virtual public Base
{
  Mid()
  {
  }
};

int main()
{
  Mid x;
  assert(x.r == 1);
  assert(x.v == 7);
  return 0;
}
