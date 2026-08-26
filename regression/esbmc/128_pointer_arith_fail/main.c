#include <assert.h>

int main()
{
  int a[8];
  int *p = a;
  __int128 i = 2;
  int *q = p + i;
  /* The offset is 2, so this must be reported, not folded away. */
  assert(q - p == 3);
  return 0;
}
