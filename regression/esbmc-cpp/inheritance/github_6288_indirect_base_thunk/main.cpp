// github #6288: virtual dispatch through an INDIRECT non-first base subobject.
// B is a direct base of Middle but only an indirect base of Derived, and sits
// at a non-zero offset (First is polymorphic, so B cannot be the primary base).
// Derived::f's thunk must adjust the B* this by B's *cumulative* offset within
// Derived, otherwise Derived::f reads `d` from the wrong location.
#include <cassert>

struct First { virtual void g() {} long pad; };
struct B { int b; virtual int f() { return b; } };
struct Middle : First, B { };
struct Derived : Middle { int d; int f() override { return d; } };

int main()
{
  Derived x;
  x.d = 42;
  x.b = 7;
  B *p = &x;
  assert(p->f() == 42);
  return 0;
}
