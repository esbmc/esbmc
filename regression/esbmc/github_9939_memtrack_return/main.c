/* #9939: a memtrack target points at the return the path left main by. */
#include <stdlib.h>
int __VERIFIER_nondet_int(void);

int main(void)
{
  int *p = malloc(sizeof(int));
  if (__VERIFIER_nondet_int())
  {
    free(p);
    return 1;
  }
  return 0;
}
