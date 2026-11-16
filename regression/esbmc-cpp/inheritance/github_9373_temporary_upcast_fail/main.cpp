// #9373: the lifetime-extended temporary still holds b == 2.
#include <cassert>
struct V { int v = 7; };
struct A { int a = 1; };
struct B { int b = 2; ~B() { b = 0; } };
struct C : A, B, virtual V { };
C makeC() { return C(); }

int main()
{
  const B &r = makeC();
  assert(r.b != 2);
  return 0;
}
