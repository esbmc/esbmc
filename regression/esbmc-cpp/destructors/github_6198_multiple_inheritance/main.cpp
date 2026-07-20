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
  // Expected 111. ~C overrides two base destructors, so
  // get_ultimate_overridden_method() (clang_cpp_convert_vft.cpp) stops at its
  // `size_overridden_methods() == 1` guard and keys the vtable slot by ~C's
  // own id instead of ~A's, leaving the slot bound to ~A. Tracked by #1866 /
  // #3894 alongside the other multiple-inheritance layout defects.
  assert(n == 111);
  return 0;
}
