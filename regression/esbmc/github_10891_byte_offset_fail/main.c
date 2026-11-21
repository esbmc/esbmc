/* #10891: a packed struct written and read at a symbolic byte offset into a uint32_t array */
#include <assert.h>
#include <stdint.h>
unsigned nondet_uint(void);
struct __attribute__((packed)) Unaligned { uint32_t v; };
int main(void)
{
  uint32_t buf[2];
  unsigned o = nondet_uint();
  __ESBMC_assume(o <= sizeof(buf) - sizeof(uint32_t) + 1);
  struct Unaligned *p = (struct Unaligned *)((unsigned char *)buf + o);
  struct Unaligned val = {7};
  *p = val;
  struct Unaligned r = *p;
  assert(r.v == 7);
  return 0;
}
