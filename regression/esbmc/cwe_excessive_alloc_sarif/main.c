#include <stdlib.h>

extern unsigned __VERIFIER_nondet_uint(void);

int main(void)
{
  unsigned n = __VERIFIER_nondet_uint();
  char *p = malloc(n);
  free(p);
  return 0;
}
