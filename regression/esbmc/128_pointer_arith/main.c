#include <assert.h>

int main()
{
  int a[8];
  int *p = a;
  __int128 i = 2;
  /* 6.5.6: additive pointer arithmetic converts neither operand, so the
     128-bit index must not flatten p to an integer. */
  int *q = p + i;
  assert(q - p == 2);
  return 0;
}
