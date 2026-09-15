// KNOWNBUG. When two sibling bases share a virtual base, the siblings' OWN
// non-virtual members read wrongly from the complete object -- no upcast
// needed, unlike the github_7025_vbase_shared_diamond residual, which needs a
// pointer to a non-first base.
//
// The failure is selective, which is the useful part:
//
//   o.a == 1   PASSED   the shared virtual base's member
//   o.b == 2   FAILED   first base's own member
//   o.c == 3   FAILED   second base's own member
//   o.d == 4   PASSED   the most-derived member
//
// so the virtual base itself lands correctly and it is B::b and C::c that are
// displaced. vbase_single_and_nonvirtual pins the neighbouring shapes that do
// work: one virtual base, a virtual base one level down, and a non-virtual
// diamond. g++ runs this program with all four assertions holding.
#include <cassert>

struct A
{
  int a = 1;
};
struct B : virtual A
{
  int b = 2;
};
struct C : virtual A
{
  int c = 3;
};
struct D : B, C
{
  int d = 4;
};

int main()
{
  D o;
  assert(o.a == 1);
  assert(o.b == 2);
  assert(o.c == 3);
  assert(o.d == 4);
  return 0;
}
