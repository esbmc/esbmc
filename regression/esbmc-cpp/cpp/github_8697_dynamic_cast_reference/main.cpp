#include <cassert>
struct A { int a = 1; virtual ~A() {} };
struct B { int b = 2; virtual ~B() {} };
struct C : A, B { };
int main()
{
  C c;
  A *pa = &c;
  B *pb = dynamic_cast<B *>(pa);
  assert(pb->b == 2);
  B &rb = dynamic_cast<B &>(*pa);
  assert(&rb == pb);
  assert(rb.b == 2);
  return 0;
}
