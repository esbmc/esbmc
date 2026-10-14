// esbmc/esbmc#3894: dynamic_cast must land on the most-derived object, so
// under the flattened virtual-base layout the ZZ subobject's displacement has
// to be subtracted. Leaving the pointer at the subobject made this assertion
// hold.
#include <cassert>

struct AA
{
  int aa;
};
struct BB
{
  int bb;
};
struct VB
{
  int vb;
};
struct ZZ
{
  virtual ~ZZ()
  {
  }
  int z;
};
struct D : AA, BB, ZZ, virtual VB
{
  int d;
};

int main()
{
  D o;
  ZZ *pz = &o;
  void *pv = dynamic_cast<void *>(pz);
  assert(pv != (void *)&o);
  return 0;
}
