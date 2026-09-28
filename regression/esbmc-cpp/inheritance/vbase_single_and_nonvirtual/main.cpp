// The controls for vbase_diamond_member_offsets: the neighbouring shapes that
// do read correctly, so the defect stays localised to a virtual base SHARED by
// two siblings rather than to virtual inheritance generally.
#include <cassert>

struct A
{
  int a = 1;
};

// one virtual base
struct B1 : virtual A
{
  int b = 2;
};

// a virtual base one level further down
struct B2 : virtual A
{
  int b = 2;
};
struct C2 : B2
{
  int c = 3;
};

// a non-virtual diamond: A is duplicated, so each subobject is separate
struct NB : A
{
  int b = 2;
};
struct NC : A
{
  int c = 3;
};
struct ND : NB, NC
{
  int d = 4;
};

int main()
{
  B1 x;
  assert(x.a == 1 && x.b == 2);

  C2 y;
  assert(y.a == 1 && y.b == 2 && y.c == 3);

  ND z;
  assert(z.NB::a == 1 && z.b == 2 && z.NC::a == 1 && z.c == 3 && z.d == 4);
  return 0;
}
