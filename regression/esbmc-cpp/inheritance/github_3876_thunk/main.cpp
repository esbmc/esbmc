// Regression test for GitHub issue #3876 (thunk this-adjustment):
// Verifies that the vtable thunk correctly adjusts this from B8* back to
// Derived* so that data member access in the overriding guard() is valid.
// Without the thunk fix, guard() reads value at the wrong address.
#include <cassert>

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
  d.value = true;
  assert(d.run());
  return 0;
}
