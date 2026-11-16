/* #9360: a[i++] has type int[m], a variable length array, so it is evaluated
   and i becomes 1. */
#include <assert.h>

int main(void)
{
  int n = 3, m = 4, i = 0;
  int a[n][m];
  (void)sizeof(a[i++]);
  assert(i == 0);
  return 0;
}
