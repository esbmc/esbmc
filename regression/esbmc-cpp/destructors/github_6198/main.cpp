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
  A *p = new C();
  delete p;
  // [expr.delete]/3, [class.dtor]/9: the virtual destructor dispatches to the
  // most-derived one, and the bases are destroyed in reverse order.
  assert(n == 3);
  assert(order[0] == 3 && order[1] == 2 && order[2] == 1);
  return 0;
}
