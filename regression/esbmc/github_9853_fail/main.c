#include <assert.h>
#include <stdint.h>
typedef struct
{
  uint32_t cfg;
  uint32_t shadow[8];
  uint32_t tail;
} state_t;
static state_t g;
static void fill(uint32_t *dst, uint32_t n, uint32_t v)
{
  for (uint32_t w = 0; w < n; w++)
    dst[w] = v + w;
}
int main(void)
{
  unsigned int k = nondet_uint();
  __ESBMC_assume(k < 4);
  g.cfg = 1;
  g.tail = 2;
  fill(&g.shadow[k], 4, 10);
  assert(g.shadow[k] == 10);
  assert(g.shadow[k + 3] == 13);
  assert(g.shadow[k + 3] == 12);
  return 0;
}
