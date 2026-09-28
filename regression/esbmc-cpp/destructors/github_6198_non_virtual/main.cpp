#include <cassert>

int n = 0;

struct A
{
  ~A()
  {
    n += 1;
  }
};
struct C : A
{
  ~C()
  {
    n += 10;
  }
};

int main()
{
  A *p = new C();
  delete p;
  // A non-virtual destructor stays statically bound: only ~A runs.
  assert(n == 1);
  return 0;
}
