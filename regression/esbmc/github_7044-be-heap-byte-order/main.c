/* Pins the byte order itself, not just the round trip: on a big-endian target
   the most significant byte of a heap int is at the lowest address. */
#include <assert.h>
#include <stdlib.h>

int main(void)
{
  int *v = (int *)malloc(sizeof(int) * 4);
  __ESBMC_assume(v);
  v[0] = 0x01020304;

  unsigned char *b = (unsigned char *)v;
  assert(b[0] == 0x01);
  assert(b[1] == 0x02);
  assert(b[2] == 0x03);
  assert(b[3] == 0x04);
  return 0;
}
