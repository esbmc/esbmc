// Regression test for GitHub issue #3876 (nondet computed guard):
// x and y are nondet; asserts that run() always equals (x > y), verifying
// that the thunk this-adjustment is correct for all possible input values.
#include <cassert>

int nondet_int();

struct B1 { virtual void dummy() = 0; };
struct B8 { virtual bool guard() = 0; bool eval() { return guard(); } };

struct Derived : B1, B8
{
  int x;
  int y;
  void dummy() override {}
  bool guard() override { return x > y; }
  bool run() { return B8::eval(); }
};

int main()
{
  Derived d;
  d.x = nondet_int();
  d.y = nondet_int();
  assert(d.run() == (d.x > d.y));
  return 0;
}
