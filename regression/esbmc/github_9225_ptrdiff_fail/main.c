/* github #9225: a negative difference that is not a whole number of elements
 * truncates toward zero, so -6/4 is -1 and not -2. */
#include <assert.h>

int a[10];

int main(void)
{
  int *q = (int *)((char *)a + 6);
  assert(a - q == -2);
  return 0;
}
