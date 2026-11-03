#include <cassert>
struct A { int a = 1; };
struct B { int b = 2; };
struct C : A, B { };
C c;
int main()
{
  B *got = nullptr;
  try { throw &c; } catch (B *pb) { got = pb; }
  assert(got == static_cast<B *>(&c));
  assert(got->b == 2);
  try { throw (C *)nullptr; } catch (B *pb) { got = pb; }
  assert(got == nullptr);
  A *ga = nullptr;
  try { throw &c; } catch (A *pa) { ga = pa; }
  assert(ga->a == 1);
  return 0;
}
