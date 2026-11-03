#include <assert.h>
#include <string.h>

/* The little-endian reading of big_endian_memset. */
int main()
{
  unsigned x = 0;
  memset(&x, 0xff, 1);
  assert(x == 0xffu);
  return 0;
}
