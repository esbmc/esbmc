/* The issue's reproducer: the write stores buf[3] (#11013). */
#include <assert.h>
union u { unsigned char t; unsigned char b[4]; };
int main(void)
{
  unsigned char buf[4] __attribute__((aligned(4))) = {0, 0, 0, 0};
  union u n = {.b = {1, 2, 3, 4}};
  *(union u *)buf = n;
  assert(buf[3] != 4);
  return 0;
}
