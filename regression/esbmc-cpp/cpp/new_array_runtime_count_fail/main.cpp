// An array new with a nondeterministic element count: the object's type names
// the count, and every read of the object must see the same type.
#include <cassert>
struct S
{
  int a, b;
};
int nondet_int();
int main()
{
  int n = nondet_int();
  __ESBMC_assume(n >= 2 && n <= 4);
  S *p = new S[n]();
  if (n > 3)
    p[0].b = 2;
  assert(p[0].b != 2);
  delete[] p;
  return 0;
}
