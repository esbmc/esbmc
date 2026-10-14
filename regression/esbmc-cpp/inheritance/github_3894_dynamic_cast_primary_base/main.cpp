// esbmc/esbmc#3894: dynamic_cast re-bases the source pointer off its static
// subobject onto the runtime type and then onto the target subobject. Both
// displacements must come from ESBMC's layout, not clang's ASTRecordLayout,
// which puts the primary base B at offset 0 while ESBMC does not.
#include <cassert>

struct A
{
  int a;
};
struct B
{
  virtual ~B()
  {
  }
  int b;
};
struct D : A, B
{
  int d;
};

int main()
{
  D o;
  o.a = 1;
  o.b = 2;
  o.d = 3;
  B *pb = &o;

  D *pd = dynamic_cast<D *>(pb);
  assert(pd != 0);
  assert(pd->d == 3);
  assert(pd->a == 1);

  void *pv = dynamic_cast<void *>(pb);
  assert(pv == (void *)&o);
  return 0;
}
