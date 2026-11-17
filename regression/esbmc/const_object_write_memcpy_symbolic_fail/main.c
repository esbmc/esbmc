#include <stdio.h>
#include <string.h>
const int t[2] = {1, 2};
const int c = 3;
void __VERIFIER_nondet_memory(void *, size_t);
int main()
{
  int s[2] = {7, 8};
  unsigned n = nondet_uint();
  __ESBMC_assume(n > 0 && n <= sizeof t);
  memcpy((void *)t, s, n);
  return 0;
}
