#include <stdlib.h>

extern unsigned __VERIFIER_nondet_uint(void);

int main(void)
{
  // The size is bounded well below the default 1 MiB
  // --excessive-alloc-check limit, so no path can reach an excessive
  // allocation: verification succeeds silently.
  unsigned n = __VERIFIER_nondet_uint();
  if (n <= 1024)
  {
    char *p = malloc(n);
    free(p);
  }
  return 0;
}
