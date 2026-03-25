// Regression test for GitHub issue #3876 (expected failure with nondet value):
// value is nondet; asserts that run() is always true without any assumption.
// This must fail: when value == false, run() returns false and the assertion
// is violated. Verifies that ESBMC finds the counterexample correctly after
// the thunk this-adjustment fix.
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
  Derived d;
  d.value = nondet_bool();
  assert(d.run()); // fails when value == false
  return 0;
}
