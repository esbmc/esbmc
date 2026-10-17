#include <assert.h>
#include <stdint.h>
#include <string.h>

/* On a big-endian target the byte at the lowest address is the most
 * significant one, so a partial memset fills a scalar from the top. */
struct S
{
  unsigned short h;
  unsigned char c;
};

int main()
{
  unsigned x = 0;
  memset(&x, 0xff, 1);
  assert(x == 0xff000000u);

  unsigned y = 0;
  memset((char *)&y + 1, 0xab, 1);
  assert(y == 0x00ab0000u);

  unsigned short a[2] = {0, 0};
  memset(a, 0x11, 3);
  assert(a[0] == 0x1111 && a[1] == 0x1100);

  unsigned b[2] = {0, 0};
  memset((char *)b + 1, 0x33, 4);
  assert(b[0] == 0x00333333u && b[1] == 0x33000000u);

  struct S s = {0, 0};
  memset(&s, 0x22, 1);
  assert(s.h == 0x2200);

  int *p = 0;
  memset(&p, 0xff, 1);
  assert((uintptr_t)p == 0xff00000000000000ull);
  return 0;
}
