/* A union whose first member is narrower than itself, read from a byte
 * array (#11013). */
#include <assert.h>
union u { unsigned char t; unsigned char b[4]; };
int main(void)
{
  unsigned char buf[4] __attribute__((aligned(4))) = {1, 2, 3, 4};
  union u v = *(union u *)buf;
  assert(v.b[3] == 4);
  return 0;
}
