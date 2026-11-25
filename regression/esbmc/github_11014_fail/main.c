/* #11014: --ir reads and writes a bit-field struct through another struct */
#include <assert.h>
struct d { unsigned a : 4, b : 4, c : 24; };
struct u { unsigned char b[4]; };
int main(void)
{
  struct d d = {1, 2, 0x332211};
  struct u *p = (struct u *)&d;
  struct u v = *p;
  assert(v.b[0] == 0x21 && v.b[1] == 0x11 && v.b[3] == 0x33);
  struct u w = {{0x65, 0x44, 0x55, 0x66}};
  *p = w;
  assert(d.a == 5 && d.b == 6 && d.c == 0x665545);
  return 0;
}
