/* A multi-byte value written to heap storage is stitched from a byte array,
   and the write used to scatter the bytes in the opposite order to the read
   under --big-endian, so the value did not survive the round trip. */
#include <assert.h>
#include <stdlib.h>

int main(void)
{
  int *v = (int *)malloc(sizeof(int) * 15);
  __ESBMC_assume(v);
  v[0] = 1;
  assert(v[0] == 1);
  v[14] = -7;
  assert(v[14] == -7);
  return 0;
}
