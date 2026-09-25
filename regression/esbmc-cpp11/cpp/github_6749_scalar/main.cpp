#include <cassert>
struct B
{
  virtual int f(int v);
};
int B::f(int v)
{
  return v;
}
int main()
{
  B b;
  B *p = &b;
  assert(p->f(7) == 7);
  return 0;
}
