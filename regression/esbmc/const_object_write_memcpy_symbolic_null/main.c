#include <string.h>
const int t[2] = {1, 2};
int main()
{
  int s[2] = {7, 8};
  int c = nondet_int();
  int *p = c ? (int *)t : NULL;
  __ESBMC_assume(!c);
  unsigned n = nondet_uint();
  __ESBMC_assume(n > 0 && n <= sizeof t);
  memcpy(p, s, n);
  return 0;
}
