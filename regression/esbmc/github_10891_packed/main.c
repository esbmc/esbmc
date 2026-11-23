/* #10891: a packed struct over a scalar array at a symbolic offset. */
#include <assert.h>
#include <stdint.h>
unsigned nondet_uint(void);
struct __attribute__((packed)) S { uint32_t v; };
int main(void)
{
  uint32_t buf[2] = {7, 7};
  unsigned i = nondet_uint();
  __ESBMC_assume(i < 2);
  const struct S *p = (const struct S *)&buf[i];
  struct S s = *p;
  assert(s.v == 7);
  return 0;
}
