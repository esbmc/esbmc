// #9373: pb points at the B subobject of gc, so pb->b == 2.
#include <cassert>
struct V { int v = 7; };
struct A { int a = 1; };
struct B { int b = 2; };
struct C : A, B, virtual V { };
C gc;
C *get() { return &gc; }

int main()
{
  B *pb = get();
  assert(pb->b != 2);
  return 0;
}
