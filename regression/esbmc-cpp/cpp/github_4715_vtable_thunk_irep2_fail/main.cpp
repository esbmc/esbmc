// esbmc/esbmc#4715: the vtable type symbols are stored IREP2-side, so a thunk's
// symbol name comes back through the migrate seam. Dispatching through the
// second base needs a this-adjusting thunk, whose name the builder takes from
// the vtable component's base name.
#include <cassert>

struct A
{
  virtual ~A()
  {
  }
  virtual int a()
  {
    return 1;
  }
};

struct B
{
  virtual ~B()
  {
  }
  virtual int b()
  {
    return 2;
  }
};

struct C : A, B
{
  int a() override
  {
    return 10;
  }
  int b() override
  {
    return 20;
  }
};

int main()
{
  C c;
  A *pa = &c;
  B *pb = &c;
  assert(pa->a() == 10);
  assert(pb->b() == 21);
  return 0;
}
