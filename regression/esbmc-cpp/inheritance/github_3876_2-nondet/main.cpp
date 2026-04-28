// Regression test for GitHub issue #3876 (nondet pointer selection):
// A nondet bool selects between two Derived objects (one with value=true,
// one with value=false). Asserts that p->run() always equals p->value,
// verifying that this-adjustment is correct regardless of which object is
// dispatched through.
#include <cassert>

bool nondet_bool();

struct B1 { virtual void dummy() = 0; };
struct B8 { virtual bool guard() = 0; bool eval() { return guard(); } };

struct Derived : B1, B8
{
  bool value;
  void dummy() override {}
  bool guard() override { return value; }
  bool run() { return B8::eval(); }
};

int main()
{
  Derived d1, d2;
  d1.value = true;
  d2.value = false;
  Derived *p = nondet_bool() ? &d1 : &d2;
  assert(p->run() == p->value);
  return 0;
}
