#include <cassert>

int nondet_int();

int main()
{
  int *p = new int[4]();
  assert(p[0] == 0 && p[3] == 0);
  delete[] p;

  int n = nondet_int();
  __ESBMC_assume(n > 0 && n < 4);
  int *q = new int[n]();
  assert(q[0] == 0);
  delete[] q;

  int *r = new int[2]{};
  assert(r[0] == 0 && r[1] == 0);
  delete[] r;

  return 0;
}
