/* #9939: the leaking path jumps past the first return to the second. */
#include <stdlib.h>
int __VERIFIER_nondet_int(void);

int main(void)
{
  int *p = malloc(sizeof(int));
  if (__VERIFIER_nondet_int())
    goto out;
  free(p);
  return 1;
out:
  return 0;
}
