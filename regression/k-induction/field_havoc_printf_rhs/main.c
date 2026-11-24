#include <assert.h>
#include <stdio.h>
struct state_t
{
  int cfg;
  unsigned hdr[8];
};
static struct state_t state;
unsigned nondet_uint(void);
static void fill(unsigned *dest)
{
  *dest = 0;
}
int main(void)
{
  unsigned n = nondet_uint();
  __ESBMC_assume(n <= 1000);
  state.cfg = 0;
  for (unsigned i = 0; i < n; ++i)
  {
    fill(&state.hdr[0]);
    if (i == 0)
      state.hdr[1] = printf("x%n", &state.cfg);
    if (i == 100)
      assert(state.cfg == 0);
  }
}
