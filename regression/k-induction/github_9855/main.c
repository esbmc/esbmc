#include <assert.h>
#include <stdint.h>
unsigned int __VERIFIER_nondet_uint(void);
typedef struct { uint32_t offset; uint32_t size; } buffer_info_t;
void unreachable_update(buffer_info_t *const buf, uint32_t n, uint32_t base)
{
  for (uint32_t id = 0u; id < n; id++) { buf[id].offset = base + id; }
}
static uint32_t g_counter;
int main(void)
{
  uint32_t n = __VERIFIER_nondet_uint(); __ESBMC_assume(n <= 1000000u);
  g_counter = 0u;
  for (uint32_t i = 0u; i < n; i++) { g_counter = (g_counter + 1u) & 7u; assert(g_counter < 8u); }
  return 0;
}
