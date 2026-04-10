// Regression test for GitHub issue #3876 (bug variant):
// Verifies that ESBMC correctly detects a false assertion in a multiple
// inheritance scenario where virtual dispatch through the second base class
// yields false, but the assertion expects true.
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
    return false;
  }
  bool run()
  {
    return B8::eval();
  }
};

int main()
{
  Derived d;
  assert(d.run()); // guard() returns false, so run() returns false -> assertion fails
  return 0;
}
