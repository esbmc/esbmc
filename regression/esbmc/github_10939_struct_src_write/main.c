/* A union written over a struct of scalars at a dynamic (but zero) offset,
 * as ptr::swap_chunk writes it back (#10939). */
#include <assert.h>
#include <stddef.h>
struct cannot_drop { unsigned int c; };
union chunk { unsigned char bytes[4]; };
size_t nondet_size(void);
int main(void)
{
  struct cannot_drop x = {0};
  union chunk u = {{0x44, 0x33, 0x22, 0x11}};
  size_t off = nondet_size();
  __ESBMC_assume(off == 0);
  *(union chunk *)((unsigned char *)&x + off) = u;
  assert(x.c == 0x11223344u);
  return 0;
}
