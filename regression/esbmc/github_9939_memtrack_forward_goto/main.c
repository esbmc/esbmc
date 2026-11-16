/* #9939: the path follows the jump past the else branch to the last return. */
#include <stdlib.h>
int __VERIFIER_nondet_int(void);
int main(void)
{
  int *p = malloc(sizeof(int));
  if (__VERIFIER_nondet_int())
    p = 0;
  else
  {
    free(p);
    return 1;
  }
  return 0;
}
