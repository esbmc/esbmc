#include <stdlib.h>

extern unsigned __VERIFIER_nondet_uint(void);

int main(void)
{
  // Bounded VLA request: n <= 512 => sizeof(int[n]) <= 2048 bytes, well under
  // the 1 MiB default. This also guards the symbolic scaling against
  // over-counting: a spurious n*n byte term (16 MiB at n=512) would push the
  // request past the bound and wrongly FAIL.
  unsigned n = __VERIFIER_nondet_uint();
  if (n <= 512)
  {
    int *p = malloc(sizeof(int[n]));
    free(p);
  }
  return 0;
}
