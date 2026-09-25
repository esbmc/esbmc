#include <stdlib.h>

extern void __VERIFIER_assume(int);

int main(void)
{
  char *p = malloc(0);
  __VERIFIER_assume((unsigned long)p != (unsigned long)0);
  *p = 1;
  return 0;
}
