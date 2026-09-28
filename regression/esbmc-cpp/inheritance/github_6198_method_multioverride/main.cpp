// github #6198: a single derived method overriding the *same signature* from
// two independent bases must override BOTH base vtable slots. Previously the
// per-base thunks inherited the derived method's virtual_name (which, for a
// multi-override, get_ultimate_overridden_method keys by the derived method
// itself), so neither base slot was overridden and dispatch through a base
// pointer wrongly called the original base method.
#include <cassert>

int n = 0;

struct A { virtual void f() { n = 1; } };
struct B { virtual void f() { n = 10; } };
struct C : A, B { void f() override { n = 100; } };

int main()
{
  C c;
  A *pa = &c;
  B *pb = &c;
  pa->f();          // through A's vtable -> C::f
  assert(n == 100);
  pb->f();          // through B's vtable -> C::f
  assert(n == 100);
  return 0;
}
