/* A union whose first member is narrower than itself, read and written over
 * a struct of scalars: it is built through its full-size member (#10939). */
#include <assert.h>
struct d { unsigned c; };
union u { unsigned char t; unsigned char b[4]; };
int main(void)
{
  struct d d = {0x11223344u};
  union u v = *(union u *)&d;
  union u n = {.b = {1, 2, 3, 4}};
  *(union u *)&d = n;
  assert(v.b[3] == 0x11 && d.c == 0x04030201u);
  return 0;
}
