#include <cassert>

struct P1
{
  int a;
  P1() : a(1) {}
};

struct P2
{
  int b;
  P2() : b(2) {}
};

struct A : P1, P2
{
  int x;
  A() : x(10) {}
  virtual ~A() {}
};

struct Mid
{
  int m;
  Mid() : m(3) {}
};

// A padding base precedes the inherited one at every level, so each hop of the
// A -> B -> C chain carries its own non-zero displacement. The downcast has to
// undo the whole sum, not just the last hop (#1866).
struct B : Mid, A
{
  int y;
  B() : y(20) {}
};

struct C : B
{
  int z;
  C() : z(30) {}
};

int main()
{
  C c;

  A *pa = &c;
  assert(pa->x == 10);

  C *pc = static_cast<C *>(pa);
  assert(pc == &c);
  assert(pc->z == 30);
  assert(pc->y == 20);
  assert(pc->m == 3);

  // Stopping at the intermediate class must land on the same B subobject.
  B *pb = static_cast<B *>(pa);
  assert(pb->y == 20);
  assert(pb == static_cast<B *>(&c));

  // A write through the downcast pointer is observable through the original.
  pc->z = 99;
  assert(c.z == 99);

  // [expr.static.cast]/11: a null operand yields a null pointer rather than
  // the displaced (char *)0 - offset.
  A *pnull = 0;
  C *cnull = static_cast<C *>(pnull);
  assert(cnull == 0);

  return 0;
}
