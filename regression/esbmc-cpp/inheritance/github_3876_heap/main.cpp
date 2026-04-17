// Regression test for GitHub issue #3876 (heap-allocation variant):
// Verifies that heap-allocated Derived objects still dispatch correctly
// via the B8 sub-object, and that 'delete p' succeeds (the this-adjustment
// in the DerivedToBase cast must NOT fire for 'new Derived()' assignments,
// only for method-receiver casts).
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
  Derived *p = new Derived();
  p->value = true;
  assert(p->run());
  delete p;
  return 0;
}
