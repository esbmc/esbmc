// Regression test for GitHub issue #3876 (lvalue variant):
// Verifies that virtual dispatch works when the Derived object is used as
// an lvalue (not a pointer), i.e. 'd.B8::eval()' rather than 'ptr->eval()'.
// The DerivedToBase cast here is lvalue-to-lvalue (struct type, not pointer),
// so the byte-offset adjustment must NOT fire (it would produce invalid IR).
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
