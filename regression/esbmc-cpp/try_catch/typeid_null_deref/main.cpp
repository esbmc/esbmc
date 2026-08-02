// C++ [expr.typeid]/2: typeid applied to *p, where p is a null pointer and the
// type is polymorphic, evaluates the operand and must fail (the standard
// mandates std::bad_typeid). ESBMC reports the null dereference -> FAILED.
#include <typeinfo>
struct Poly { virtual void m() {} };
int main()
{
  Poly *p = 0;
  const char *n = typeid(*p).name();
  return n ? 0 : 1;
}
