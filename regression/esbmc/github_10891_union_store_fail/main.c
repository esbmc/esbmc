/* #10891 */
#include <assert.h>
#include <stdint.h>
unsigned nondet_uint(void);
union MU { uint32_t value; char uninit; };
struct __attribute__((packed)) Unaligned { uint32_t v; };
int main(void)
{
  union MU buf[2];
  unsigned o = nondet_uint();
  __ESBMC_assume(o <= sizeof(buf) - sizeof(uint32_t));
  struct Unaligned *p = (struct Unaligned *)((unsigned char *)buf + o);
  struct Unaligned val = {7};
  *p = val;
  struct Unaligned r = *p;
  assert(r.v == 8);
  return 0;
}
