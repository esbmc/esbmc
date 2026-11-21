/* #10891: a packed struct read over a union array element at a symbolic index */
#include <assert.h>
#include <stdint.h>
unsigned nondet_uint(void);
union MU { uint32_t value; char uninit; };
struct __attribute__((packed)) Unaligned { uint32_t v; };
int main(void)
{
  union MU buf[2] = {{7}, {7}};
  unsigned i = nondet_uint();
  __ESBMC_assume(i < 2);
  struct Unaligned r = *(struct Unaligned *)&buf[i];
  assert(r.v == 7);
  return 0;
}
