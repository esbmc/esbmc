// Regression test for GitHub issue #3876 (explicit static_cast variant):
// Verifies that virtual dispatch works when the base-class pointer is formed
// via an explicit static_cast<B8*>(ptr) rather than an implicit conversion.
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
  Derived *ptr = &d;
  assert(static_cast<B8 *>(ptr)->eval());
  return 0;
}
