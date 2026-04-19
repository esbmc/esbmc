// Regression test for GitHub issue #3876 (two subclasses with different B8 offsets):
// DerivedA : B1, B8 places B8 at offset sizeof(B1).
// DerivedB : B1, B2, B8 places B8 at offset sizeof(B1)+sizeof(B2).
// Both classes have nondet guard values. Asserts that run() always matches
// the stored field in each class, exercising two distinct thunk adjustments.
#include <cassert>

bool nondet_bool();

struct B1 { virtual void dummy1() = 0; };
struct B2 { virtual void dummy2() = 0; };
struct B8 { virtual bool guard() = 0; bool eval() { return guard(); } };

struct DerivedA : B1, B8
{
  bool va;
  void dummy1() override {}
  bool guard() override { return va; }
  bool run() { return B8::eval(); }
};

struct DerivedB : B1, B2, B8
{
  bool vb;
  void dummy1() override {}
  void dummy2() override {}
  bool guard() override { return vb; }
  bool run() { return B8::eval(); }
};

int main()
{
  DerivedA a;
  DerivedB b;
  a.va = nondet_bool();
  b.vb = nondet_bool();
  assert(a.run() == a.va);
  assert(b.run() == b.vb);
  return 0;
}
