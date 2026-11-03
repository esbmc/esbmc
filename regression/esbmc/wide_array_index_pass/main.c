#include <assert.h>

typedef unsigned _BitInt(128) index_t;
__attribute__((annotate("__ESBMC_inf_size:128"))) extern unsigned values[1];

int main(void)
{
  index_t high = (index_t)1 << 80;
  values[0] = 11;
  values[high] = 22;
  assert(values[0] == 11 && values[high] == 22);
  return 0;
}
