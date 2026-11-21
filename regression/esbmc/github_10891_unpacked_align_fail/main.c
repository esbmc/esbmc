/* #10891 */
#include <stdint.h>
unsigned nondet_uint(void);
struct N { uint32_t v; };
int main(void)
{
  uint32_t buf[2] = {1, 2};
  unsigned o = nondet_uint();
  __ESBMC_assume(o <= 4);
  struct N r = *(struct N *)((unsigned char *)buf + o);
  return r.v;
}
