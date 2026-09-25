// Negative companion to github_6198_method_multioverride: dispatch through the
// second base B* must reach C::f (n == 100), so asserting the pre-fix wrong
// value (the original B::f side effect, n == 10) is violated.
#include <cassert>

int n = 0;

struct A { virtual void f() { n = 1; } };
struct B { virtual void f() { n = 10; } };
struct C : A, B { void f() override { n = 100; } };

int main()
{
  C c;
  B *pb = &c;
  pb->f();
  assert(n == 10); // wrong on purpose: the fix makes this C::f, so n == 100
  return 0;
}
