#include <cassert>
struct B
{
  int x;
  B() throw() : x(1)
  {
  }
  virtual int f()
  {
    return 1;
  }
  virtual ~B() throw()
  {
  }
};
struct D : B
{
  D() throw()
  {
  }
  int f()
  {
    return 2;
  }
};
int main()
{
  D d;
  B *p = &d;
  assert(p->f() == 2);
  assert(d.x == 1);
  return 0;
}
