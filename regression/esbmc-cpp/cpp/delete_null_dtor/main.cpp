#include <cassert>

int destroyed = 0;

struct A
{
  virtual ~A()
  {
    destroyed++;
  }
};

int main()
{
  A *p = nullptr;
  delete p;
  assert(destroyed == 0);

  A *q = new A();
  delete q;
  assert(destroyed == 1);
  return 0;
}
