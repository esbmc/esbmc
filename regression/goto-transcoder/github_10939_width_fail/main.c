/* A union wider than the scalar it is read over: still refused at a dynamic
 * zero offset, as it is at a constant one. */
#include <assert.h>
#include <stddef.h>
union chunk { unsigned char bytes[8]; };
size_t nondet_size(void);
int main(void)
{
  unsigned int x = 0x11223344u;
  size_t off = nondet_size();
  __CPROVER_assume(off == 0);
  union chunk *p = (union chunk *)((unsigned char *)&x + off);
  union chunk a = *p;
  assert(a.bytes[0] == 0x44);
  return 0;
}
