/* As github_10939_narrow_first, but asserting the write left a byte out. */
#include <assert.h>
struct d { unsigned c; };
union u { unsigned char t; unsigned char b[4]; };
int main(void)
{
  struct d d = {0x11223344u};
  union u n = {.b = {1, 2, 3, 4}};
  *(union u *)&d = n;
  assert(d.c != 0x04030201u);
  return 0;
}
