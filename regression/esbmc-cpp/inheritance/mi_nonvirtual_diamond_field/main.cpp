// Non-virtual diamond: B and C each (non-virtually) inherit A, and D inherits
// both B and C. D therefore contains TWO distinct A subobjects, so there are
// two distinct A::a fields (one under B, one under C). ESBMC currently
// flattens D and deduplicates base components by name, collapsing the two
// A::a into one field — so writes through the two paths alias, and a read of
// A::a through a C& upcast returns the wrong value.
//
// Ground truth (g++): the two A subobjects are independent. After setting
// the C-path A::a to 3 and D::d to 99, a C& binds the C subobject and
// cc.a == 3 (NOT 99, NOT whatever the B-path A::a holds).
#include <cassert>

struct A
{
  int a;
};
struct B : A
{
  int b;
};
struct C : A
{
  int c;
};
struct D : B, C
{
  int d;
};

int main()
{
  D d;
  d.B::a = 1; // A::a reached through the B subobject
  d.C::a = 3; // A::a reached through the C subobject (a distinct field)
  d.d = 99;

  // The two A subobjects must hold independent values.
  assert(d.B::a == 1);
  assert(d.C::a == 3);

  // Read A::a through a reference to the C subobject.
  C &cc = d;
  assert(cc.a == 3);

  return 0;
}
