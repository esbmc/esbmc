// github.com/esbmc/esbmc/issues/7643: resolving a record type to its defining
// declaration lets a field's pointee definition re-enter the record being
// converted. Completing it twice used to add the vtable symbol twice.
#include <cassert>

struct A
{
  struct B;
  B *p;
  virtual void f()
  {
  }
  int x;
};

struct A::B : A
{
  int y;
};

int main()
{
  A::B b;
  b.x = 1;
  b.y = 2;
  b.p = &b;
  assert(b.p->x == 1);
  return 0;
}
