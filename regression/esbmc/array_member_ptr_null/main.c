#include <assert.h>
#include <stdint.h>
#include <stdlib.h>
#define WORDS 4u
typedef struct
{
  uint32_t f0;
  uint32_t f1;
  uint32_t hdr_shadow[3 * WORDS];
} state_t;
static state_t *g_heap;
static void dma_read(uint32_t *dst, uint32_t nwords)
{
  for (uint32_t w = 0; w < nwords; w++)
    dst[w] = nondet_uint();
}
int main(void)
{
  g_heap = malloc(sizeof(state_t));
  if (!g_heap)
    return 0;
  uint32_t n = nondet_uint();
  __ESBMC_assume(n <= 1000000u);
  for (uint32_t i = 0; i < n; i++)
  {
    for (uint32_t page = 0; page < 3u; page++)
      dma_read(&g_heap->hdr_shadow[page * WORDS], WORDS);
    assert(i < 1000000u);
  }
  return 0;
}
