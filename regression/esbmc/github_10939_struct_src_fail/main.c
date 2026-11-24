/* As github_10939_struct_src, but asserting the high byte is first. */
#include <assert.h>
#include <stddef.h>
struct cannot_drop { unsigned int c; };
union chunk { unsigned char bytes[4]; };
size_t nondet_size(void);
int main(void)
{
  struct cannot_drop x = {0x11223344u};
  size_t off = nondet_size();
  __ESBMC_assume(off == 0);
  union chunk a = *(union chunk *)((unsigned char *)&x + off);
  assert(a.bytes[0] == 0x11);
  return 0;
}
