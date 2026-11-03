// esbmc/esbmc#9376: [conv.ptr]/3 - a null derived pointer converts to a null
// pointer to a non-first base, and a non-null one to that base subobject.
#include <cassert>

struct A
{
  int a = 1;
};
struct B
{
  int b = 2;
};
struct C : A, B
{
};
struct D : C
{
};

bool nondet_bool();

int main()
{
  C c;
  C *pc = nondet_bool() ? &c : nullptr;
  B *pb = pc;
  if (pb)
    assert(pb->b == 2 && pc == &c);
  else
    assert(pc == nullptr);

  B *ps = static_cast<B *>(pc);
  assert((ps == nullptr) == (pc == nullptr));

  D d;
  D *pd = nondet_bool() ? &d : nullptr;
  B *pdb = pd;
  assert((pdb == nullptr) == (pd == nullptr));
  if (pdb)
    assert(pdb->b == 2);
  return 0;
}
