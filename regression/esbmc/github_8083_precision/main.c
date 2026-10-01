#include <stdio.h>
#include <string.h>
#include <assert.h>

/* scanf's format string and memcmp's operands are not written, so the
   interval of y survives both calls and the analysis still proves y == 3. */
int main()
{
  int x = 5, y = 3, z = 4;
  scanf("%d", &x);
  assert(y == 3);
  if (memcmp(&x, &z, sizeof x) == 0)
    x = 0;
  assert(y == 3);
  return 0;
}
