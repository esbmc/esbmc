#include <stdio.h>
#include <assert.h>

/* The interval of x is forgotten at scanf, not the program's other facts. */
int main()
{
  int x = 5;
  scanf("%d", &x);
  __ESBMC_assume(x == 5);
  assert(x == 5);
  return 0;
}
