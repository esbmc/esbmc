// typeid on a valid polymorphic object (both via a non-null dereference and a
// plain lvalue) must not fault.
#include <typeinfo>
#include <cassert>
struct Poly { virtual void m() {} };
int main()
{
  Poly obj;
  Poly *p = &obj;
  const char *n1 = typeid(*p).name();
  const char *n2 = typeid(obj).name();
  assert(n1 != 0 && n2 != 0);
  return 0;
}
