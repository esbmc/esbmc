#include <assert.h>
#include <stddef.h>
#include <stdint.h>
typedef struct { uint32_t cfg; uint32_t shadow[8]; uint32_t tail; } state_t;
static state_t g;
int main(void)
{
  unsigned k = nondet_uint();
  __ESBMC_assume(k < 8);
  g.tail = 0;
  uint32_t *p = &g.shadow[k];
  uint32_t *q = p + 0;
  char *c = (char *)q;
  state_t *st = (state_t *)(c - offsetof(state_t, shadow) - k * 4);
  st->tail = 42;
  assert(g.tail == 42);
  return 0;
}
