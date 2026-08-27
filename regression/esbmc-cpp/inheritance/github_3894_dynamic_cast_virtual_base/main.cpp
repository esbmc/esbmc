// esbmc/esbmc#3894: a hierarchy with a virtual base keeps the legacy
// flattened layout, so the dynamic_cast re-basing has no "@base@" component
// to undo. The displacement must then come from where the base's members
// landed in the flattened struct -- ZZ is the alphabetically last base, so it
// sits past AA and BB and the unadjusted pointer names the wrong object.
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
  o.aa = 1;
  o.bb = 2;
  o.z = 3;
  o.d = 4;
  o.vb = 5;

  ZZ *pz = &o;
  D *pd = dynamic_cast<D *>(pz);
  assert(pd != 0);
  assert(pd == &o);
  assert(pd->d == 4);
  assert(pd->aa == 1);

  void *pv = dynamic_cast<void *>(pz);
  assert(pv == (void *)&o);
  return 0;
}
