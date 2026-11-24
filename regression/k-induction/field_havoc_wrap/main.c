#include <assert.h>
#include <stdint.h>

typedef struct
{
  unsigned cfg;
  unsigned hdr[8];
} state_t;

static state_t state;
unsigned nondet_uint(void);

int main(void)
{
  unsigned n = nondet_uint();
  __ESBMC_assume(n <= 1000000u);
  state.cfg = 0;
  unsigned *dest = &state.hdr[0];
  for (unsigned i = 0; i < n; ++i)
  {
    if (i == 0)
    {
      dest[0x3FFFFFFFFFFFFFFFull] = 1;
    }
    if (i == 100)
      assert(state.cfg == 0);
  }
}
