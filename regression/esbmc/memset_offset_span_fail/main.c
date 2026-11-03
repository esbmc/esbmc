#include <assert.h>
#include <string.h>

/* memset_offset_span with the second element left untouched. */
int main()
{
  unsigned a[2] = {0, 0};
  memset((char *)a + 3, 0x11, 2);
  assert(a[1] == 0);
  return 0;
}
