// #9373: a materialized temporary bound to a non-first base stays alive for
// the reference and for the call on it; ~B would overwrite b.
#include <cassert>
struct V { int v = 7; };
struct A { int a = 1; };
struct B { int b = 2; int getb() const { return b; } ~B() { b = 0; } };
struct C : A, B, virtual V { };
C makeC() { return C(); }

int main()
{
  const B &r = makeC();
  assert(r.b == 2);
  assert(makeC().getb() == 2);
  return 0;
}
