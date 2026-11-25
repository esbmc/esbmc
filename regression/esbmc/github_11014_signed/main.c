/* #11014: signed bit-fields keep their two's complement bits under --ir */
#include <assert.h>
struct d { int a : 4; unsigned b : 4; int c : 24; };
struct u { unsigned char b[4]; };
int nondet_int(void);
int main(void)
{
  struct d d = {-3, 2, -2};
  struct u v = *(struct u *)&d;
  assert(v.b[0] == 0x2d && v.b[1] == 0xfe && v.b[2] == 0xff && v.b[3] == 0xff);
  struct u w = {{0x9f, 0x01, 0x00, 0x80}};
  *(struct u *)&d = w;
  assert(d.a == -1 && d.b == 9 && d.c == -8388607);
  int n = nondet_int();
  __ESBMC_assume(n > -8 && n < 8);
  d.a = n;
  v = *(struct u *)&d;
  assert((v.b[0] & 0xf) == (n & 0xf));
  return 0;
}
