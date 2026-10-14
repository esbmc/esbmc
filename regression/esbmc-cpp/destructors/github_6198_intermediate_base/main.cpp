#include <cassert>

int order[8];
int n = 0;

struct A
{
  virtual ~A()
  {
    order[n++] = 1;
  }
};
struct B : A
{
  ~B()
  {
    order[n++] = 2;
  }
};
struct C : B
{
  ~C()
  {
    order[n++] = 3;
  }
};

int main()
{
  // Delete through the intermediate base, not the root.
  B *p = new C();
  delete p;
  assert(n == 3);
  assert(order[0] == 3 && order[1] == 2 && order[2] == 1);
  return 0;
}
