// esbmc/esbmc#9227: pointers copied byte by byte through a char buffer.
#include <string.h>
int x[6];
unsigned nondet_uint(void);
int main(void)
{
  int *src[2];
  unsigned char buf[sizeof src];
  int *dst[2];
  for (int i = 0; i < 2; i++)
    src[i] = &x[nondet_uint() % 6];
  memcpy(buf, src, sizeof src);
  memcpy(dst, buf, sizeof src);
  __ESBMC_assert(dst[1] == src[1], "");
  __ESBMC_assert(dst[0] != src[0], "");
  return 0;
}
