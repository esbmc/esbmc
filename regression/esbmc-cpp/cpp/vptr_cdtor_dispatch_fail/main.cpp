// Negative variant of vptr_cdtor_dispatch: a virtual call from a base-class
// constructor resolves to the base override (A::who -> 1), so asserting it
// equals the derived override (2) must fail.
#include <cassert>

int ctor_seen;

struct A
{
  virtual int who() { return 1; }
  A() { ctor_seen = who(); }
  virtual ~A() {}
};

struct B : A
{
  int who() override { return 2; }
};

int main()
{
  B b;
  assert(ctor_seen == 2); // wrong: A() dispatches to A::who, so ctor_seen == 1
  return 0;
}
