// Two unrelated bases with a SAME-NAMED data member. Under multiple
// inheritance each base is a distinct subobject at a distinct offset, so
// B::x and C::x are two separate fields. ESBMC currently flattens the derived
// class and deduplicates base components by name, collapsing B::x and C::x
// into a single field — so the write to C::x clobbers B::x and a read through
// a C& upcast returns the wrong value.
//
// Ground truth (g++): D has two int fields (B::x@0, C::x@4) plus d@8;
// after d.B::x=10; d.C::x=20, a C& binds the C subobject and cc.x == 20,
// while B::x stays 10.
#include <cassert>

struct B
{
  int x;
};
struct C
{
  int x;
};
struct D : B, C
{
  int d;
};

int main()
{
  D d;
  d.B::x = 10;
  d.C::x = 20;
  d.d = 30;

  // The two subobjects must hold independent values.
  assert(d.B::x == 10);
  assert(d.C::x == 20);

  // Reading C::x through a reference to the non-first base subobject.
  C &cc = d;
  assert(cc.x == 20);

  return 0;
}
