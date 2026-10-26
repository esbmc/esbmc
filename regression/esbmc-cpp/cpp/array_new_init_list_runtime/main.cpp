// A braced list with an element count known only at run time.
#include <cassert>
int nondet_int();
int main()
{
  int n = nondet_int();
  __ESBMC_assume(n >= 3 && n <= 5);
  int *p = new int[n]{1, 2, 3};
  assert(p[2] == 3 && p[n - 1] == (n == 3 ? 3 : 0));
  delete[] p;
  return 0;
}
