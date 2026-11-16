// #9373: a side-effecting operand converted to a non-first base of a class
// with a virtual base is displaced onto that base, and evaluated once.
#include <cassert>
struct V { int v = 7; };
struct A { int a = 1; };
struct B { int b = 2; int getb() { return b; } };
struct C : A, B, virtual V { };
C gc;
int calls = 0;
C *get() { ++calls; return &gc; }
C &getr() { ++calls; return gc; }
C *getnull() { ++calls; return nullptr; }
int rd(B *p) { return p->b; }

int main()
{
  B *pb = get();
  assert(pb->b == 2 && calls == 1);
  pb->b = 5;
  assert(gc.a == 1 && gc.b == 5);
  gc.b = 2;

  assert(rd(get()) == 2 && calls == 2);
  assert(getr().getb() == 2 && calls == 3);
  B &rb = getr();
  assert(rb.b == 2 && calls == 4);
  assert(getnull() == nullptr && calls == 5);
  B *pn = getnull();
  assert(pn == nullptr && calls == 6);

  B *pnew = new C;
  assert(pnew->b == 2);

  C arr[2];
  C *p = arr;
  B *pi = p++;
  assert(pi->b == 2 && p == arr + 1);
  return 0;
}
