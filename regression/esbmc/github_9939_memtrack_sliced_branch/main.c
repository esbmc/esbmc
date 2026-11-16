/* #9939: the slicer drops the branch, and one return is left ahead. */
#include <stdlib.h>
int __VERIFIER_nondet_int(void);
int main(void)
{
  int *p = malloc(sizeof(int));
  p = 0;
  int x = 0;
  if (__VERIFIER_nondet_int())
    x = 1;
  return x;
}
