#include <assert.h>
#include <stdint.h>
typedef struct
{
  uint32_t cfg;
  uint32_t shadow[8];
  uint32_t tail;
} state_t;
static state_t g;
int main(void)
{
  unsigned k = nondet_uint();
  _Bool c = nondet_bool();
  __ESBMC_assume(k < 8);
  uint32_t *p = c ? &g.shadow[k] : &g.tail;
  *p = 5;
  assert(c || g.tail == 5);
  assert(!c || g.shadow[k] == 5);
  return 0;
}
