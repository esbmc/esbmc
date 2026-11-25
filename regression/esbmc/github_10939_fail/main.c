#include <assert.h>
#include <stddef.h>
union chunk { unsigned char bytes[4]; };
size_t nondet_size(void);
int main(void)
{
  unsigned int x = 0x11223344u;
  size_t off = nondet_size();
  __ESBMC_assume(off < 2);
  union chunk *p = (union chunk *)((unsigned char *)&x + off);
  union chunk a = *p;
  assert(a.bytes[0] == 0x44);
  return 0;
}
