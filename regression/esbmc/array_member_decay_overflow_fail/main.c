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
  unsigned int k = nondet_uint();
  __ESBMC_assume(k < 2);
  uint32_t *p = g.shadow + k;
  p[7] = 5;
  return 0;
}
