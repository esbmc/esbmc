#include <assert.h>

void __CPROVER_array_copy(void *, const void *);

int main(void)
{
  int s[5] = {7, 7, 7, 7, 7};
  int d[3] = {0, 0, 0};
  // Extents differ: CBMC copies only the destination's extent.
  __CPROVER_array_copy(d, s);
  assert(d[2] == 7);
  return 0;
}
