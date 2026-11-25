/* #10891 */
#include <assert.h>
#include <stdint.h>
unsigned nondet_uint(void);
struct __attribute__((packed)) U { uint32_t v; };
int main(void)
{
  uint32_t buf[2] = {0, 0};
  unsigned o = nondet_uint();
  __ESBMC_assume(o == 1);
  struct U val = {0xAABBCCDDu};
  *(struct U *)((unsigned char *)buf + o) = val;
  assert(buf[0] == 0xBBCCDD00u);
  assert(buf[1] == 0x000000AAu);
  return 0;
}
