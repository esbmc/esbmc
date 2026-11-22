/* A union written over a scalar at a dynamic (but zero) offset. */
#include <assert.h>
#include <stddef.h>
union U { unsigned short h; unsigned char b[4]; };
size_t nondet_size(void);
int main(void)
{
  unsigned int x = 0;
  size_t off = nondet_size();
  __CPROVER_assume(off == 0);
  union U v;
  v.b[0] = 0x44; v.b[1] = 0x33; v.b[2] = 0x22; v.b[3] = 0x11;
  *(union U *)((unsigned char *)&x + off) = v;
  assert(x == 0x11223344u);
  return 0;
}
