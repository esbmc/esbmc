// esbmc/esbmc#4715: the vptr-init pass stores the whole constructor body
// IREP2-side, and this body carries a `new` whose initialiser is a constructor
// call. The value is nondet so the assertion cannot be constant-folded away.
#include <cassert>
struct Foo
{
  int x;
  Foo(int v) : x(v)
  {
  }
};
struct Base
{
  virtual ~Base()
  {
  }
  virtual int get()
  {
    return 0;
  }
};
int nondet_int();
struct D : Base
{
  Foo *p;
  D(int v)
  {
    p = new Foo(v);
  }
  ~D()
  {
    delete p;
  }
  int get() override
  {
    return p->x;
  }
};
int main()
{
  int v = nondet_int();
  __ESBMC_assume(v > 10 && v < 20);
  D d(v);
  Base *b = &d;
  assert(b->get() == v + 1);
  return 0;
}
