#include <assert.h>
#include <string.h>

/* The little-endian reading of big_endian_memcpy. */
int main()
{
  unsigned y = 0x01020304u;
  unsigned x = 0;
  memcpy(&x, &y, 1);
  assert(x == 0x04u);
  return 0;
}
