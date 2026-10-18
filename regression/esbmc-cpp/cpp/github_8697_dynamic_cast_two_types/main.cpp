#include <cassert>
struct A { int a = 1; virtual ~A() {} };
struct B { int b = 2; virtual ~B() {} };
struct C1 : A, B { };
struct P { long pad[4]; };
struct C2 : A, P, B { };
int main()
{
  C1 c1;
  C2 c2;
  A *p1 = &c1, *p2 = &c2;
  B *b1 = dynamic_cast<B *>(p1);
  B *b2 = dynamic_cast<B *>(p2);
  assert(b1 == static_cast<B *>(&c1));
  assert(b2 == static_cast<B *>(&c2));
  assert(b2->b == 2);
  return 0;
}
