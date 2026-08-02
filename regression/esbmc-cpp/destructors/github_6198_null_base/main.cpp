#include <cassert>

int n = 0;

struct A
{
  virtual ~A()
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
  A *p = 0;
  // [expr.delete]/7: no destructor runs, and the null guard must skip the
  // vtable read rather than dereferencing a null vptr.
  delete p;
  assert(n == 0);
  return 0;
}
