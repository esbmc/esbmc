#include <stdlib.h>

extern unsigned __VERIFIER_nondet_uint(void);

int main(void)
{
  // Attacker-controlled, unbounded allocation size (CWE-789): n can be up to
  // UINT_MAX, well above the default 1 MiB --excessive-alloc-check bound.
  unsigned n = __VERIFIER_nondet_uint();
  char *p = malloc(n);
  free(p);
  return 0;
}
