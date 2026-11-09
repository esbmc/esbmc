#include <stdlib.h>

extern int __VERIFIER_nondet_int(void);

int depth(int n)
{
  if (n > 0)
    return depth(n - 1);
  return 0;
}

int main()
{
  int *p = malloc(sizeof(int));
  depth(__VERIFIER_nondet_int());
  p = 0;
  return 0;
}
