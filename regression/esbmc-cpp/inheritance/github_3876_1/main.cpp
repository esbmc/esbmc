// Regression test for GitHub issue #3876 (computed guard variant):
// guard() returns a computed comparison (x > y) rather than a stored bool,
// verifying that the thunk this-adjustment allows correct access to multiple
// data members of the derived class.
#include <cassert>

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
  d.x = 5;
  d.y = 3;
  assert(d.run()); // 5 > 3
  return 0;
}
