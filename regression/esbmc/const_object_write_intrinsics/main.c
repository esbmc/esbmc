#include <assert.h>
#include <stdio.h>
#include <string.h>
const int t[2] = {1, 2};
int g[2];
int h;
void __VERIFIER_nondet_memory(void *, size_t);
int main()
{
  int s[2] = {7, 8};
  int *p = nondet_bool() ? (int *)t : g;
  __ESBMC_assume(p == g);
  memcpy(p, s, sizeof g);
  assert(g[1] == 8);
  memmove(p, s, sizeof g);
  sscanf("5", "%d", &h);
  __VERIFIER_nondet_memory(p, sizeof g);
  __ESBMC_init_object(g);
  memcpy((void *)t, s, 0);
  unsigned n = nondet_uint();
  __ESBMC_assume(n <= sizeof g);
  memcpy(g, s, n);
  if (n == 0)
    memcpy((void *)t, s, n);
  assert(t[0] == 1 && t[1] == 2);
  return 0;
}
