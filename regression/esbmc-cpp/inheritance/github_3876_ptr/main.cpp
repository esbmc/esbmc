// Regression test for GitHub issue #3876 (non-this pointer variant):
// Verifies that ESBMC correctly adjusts the pointer when calling a virtual
// method through a non-first base class via a local Derived* variable (not
// 'this'). The cast 'Derived* -> B8*' must be offset by sizeof(B1) so that
// virtual dispatch through B8's vtable finds Derived::guard() (which returns
// true), not B1's vtable slot.
#include <cassert>

struct B1
{
  virtual void dummy() = 0;
};

struct B8
{
  virtual bool guard() = 0;
  bool eval()
  {
    return guard();
  }
};

struct Derived : B1, B8
{
  void dummy() override {}
  bool guard() override
  {
    return true;
  }
};

int main()
{
  Derived d;
  Derived *ptr = &d;
  assert(ptr->eval()); // virtual dispatch through B8 sub-object via local ptr
  return 0;
}
