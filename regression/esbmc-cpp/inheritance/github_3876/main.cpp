// Regression test for GitHub issue #3876:
// ESBMC incorrectly handles virtual dispatch when calling a base class method
// via a qualified name (Base::method()) in the presence of multiple inheritance.
//
// B8::eval() calls the virtual guard() method; when run() calls B8::eval()
// through the B8 sub-object of a Derived instance, the virtual dispatch must
// resolve to Derived::guard() (which returns true).
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
  bool run()
  {
    return B8::eval();
  }
};

int main()
{
  Derived d;
  assert(d.run());
  return 0;
}
