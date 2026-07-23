#include <cassert>

int n = 0;

struct A
{
  virtual ~A()
  {
    n += 1;
  }
};
struct B
{
  virtual ~B()
  {
    n += 10;
  }
};
struct C : A, B
{
  ~C()
  {
    n += 100;
  }
};

int main()
{
  A *p = new C();
  delete p;
  // Expected 111: ~C runs (100) and chains to both base destructors ~A (1) and
  // ~B (10). ~C overrides two base destructors, so get_ultimate_overridden_
  // method() cannot pick a unique base; each per-base thunk is now re-keyed by
  // that base's own virtual_name so both base vtable slots dispatch to ~C's
  // deleting destructor (#6198).
  assert(n == 111);
  return 0;
}
