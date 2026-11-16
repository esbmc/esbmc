#include <assert.h>
#include <stdint.h>
unsigned int __VERIFIER_nondet_uint(void);
typedef struct { uint32_t cfg; uint32_t shadow[64]; } state_t;
static state_t g_state;
int main(void)
{
  state_t next = {0};
  uint32_t n = __VERIFIER_nondet_uint();
  __ESBMC_assume(n <= 1000000u);
  g_state.cfg = 0u;
  for (uint32_t i = 0; i < n; i++)
  {
    g_state.shadow[i & 63u] = __VERIFIER_nondet_uint();
    assert(g_state.cfg < 100u);
    next.cfg = g_state.cfg + 1u;
    g_state = next;
  }
  return 0;
}
