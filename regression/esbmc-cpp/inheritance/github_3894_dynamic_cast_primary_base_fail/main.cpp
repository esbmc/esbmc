// esbmc/esbmc#3894: dynamic_cast<void *> yields the address of the
// most-derived object, so it must equal &o rather than the address of the B
// subobject it was handed. Taking the displacement from clang's
// ASTRecordLayout left it pointing at the subobject, which made this
// assertion hold.
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
  B *pb = &o;
  void *pv = dynamic_cast<void *>(pb);
  assert(pv != (void *)&o);
  return 0;
}
