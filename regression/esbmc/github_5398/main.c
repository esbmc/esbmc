#include <stdlib.h>

extern void __VERIFIER_assume(int);

int main(void)
{
  void *p = malloc(0);
  __VERIFIER_assume((unsigned long)p != (unsigned long)0);
  free(p);
  return 0;
}
