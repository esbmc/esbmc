// The loop allocates nothing, so the leak check keeps the inductive step,
// which proves the program free of leaks for any bound.
#include <stdlib.h>
extern int __VERIFIER_nondet_int(void);

int main(void)
{
  int *p = malloc(sizeof(int));
  if (!p)
    return 0;
  *p = 0;
  while (__VERIFIER_nondet_int())
    *p = *p + 1;
  free(p);
  return 0;
}
