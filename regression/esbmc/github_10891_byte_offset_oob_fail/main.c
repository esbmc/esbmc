/* #10891 */
#include <stdint.h>
unsigned nondet_uint(void);
struct __attribute__((packed)) U { uint32_t v; };
int main(void)
{
  uint32_t buf[2] = {1, 2};
  unsigned o = nondet_uint();
  __ESBMC_assume(o <= 5);
  struct U r = *(struct U *)((unsigned char *)buf + o);
  return r.v;
}
