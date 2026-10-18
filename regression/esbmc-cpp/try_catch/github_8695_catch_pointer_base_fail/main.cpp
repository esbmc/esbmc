#include <cassert>
struct A { int a = 1; };
struct B { int b = 2; };
struct C : A, B { };
C c;
int main()
{
  B *got = nullptr;
  try { throw &c; } catch (B *pb) { got = pb; }
  assert(got->b == 1);
  return 0;
}
