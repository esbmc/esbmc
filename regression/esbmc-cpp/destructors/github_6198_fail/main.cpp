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
  A *p = new C();
  delete p;
  // ~C must run too, so this is unreachable.
  assert(n == 1);
  return 0;
}
