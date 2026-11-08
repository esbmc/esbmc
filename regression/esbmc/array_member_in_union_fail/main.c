#include <assert.h>
#include <stdint.h>
typedef struct
{
  uint32_t cfg;
  union
  {
    uint32_t a[8];
    uint64_t b[4];
  } u;
  uint32_t tail;
} state_t;
static state_t g;
int main(void)
{
  unsigned k = nondet_uint();
  __ESBMC_assume(k < 8);
  g.tail = 0;
  uint32_t *p = &g.u.a[k];
  p[8 - k] = 42; /* g.tail */
  assert(g.tail == 0);
  return 0;
}
