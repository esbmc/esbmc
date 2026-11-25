/* As github_11013_fail, at a symbolic offset into the byte array (#11013). */
#include <assert.h>
union u { unsigned char t; unsigned char b[4]; };
int main(void)
{
  unsigned char buf[8] __attribute__((aligned(4))) = {0};
  _Bool k;
  unsigned i = k ? 4 : 0;
  union u n = {.b = {1, 2, 3, 4}};
  *(union u *)(buf + i) = n;
  assert(buf[i + 3] != 4);
  return 0;
}
