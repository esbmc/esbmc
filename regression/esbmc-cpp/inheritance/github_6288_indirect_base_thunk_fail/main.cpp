// Negative companion to github_6288_indirect_base_thunk: the dispatch reaches
// Derived::f (returns d == 42), so asserting the pre-fix wrong value (B::b == 7,
// what a zero this-adjustment would surface) is violated.
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
  assert(p->f() == 7); // wrong on purpose: dispatch reaches Derived::f, so f()==42
  return 0;
}
