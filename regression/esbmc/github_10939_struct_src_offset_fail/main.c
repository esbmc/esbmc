/* A union read over a struct of scalars at offset 1 runs past the object. */
#include <stddef.h>
struct cannot_drop { unsigned int c; };
union chunk { unsigned char bytes[4]; };
size_t nondet_size(void);
int main(void)
{
  struct cannot_drop x = {0x11223344u};
  size_t off = nondet_size();
  __ESBMC_assume(off == 1);
  union chunk a = *(union chunk *)((unsigned char *)&x + off);
  return a.bytes[0];
}
