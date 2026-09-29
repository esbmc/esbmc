/* The safe twin of multi_property_incremental_final_verdict: the forward
   condition closes with no violation recorded, so the run must still end
   VERIFICATION SUCCESSFUL (esbmc/esbmc#7900). */
#include <assert.h>
int main()
{
  unsigned n = nondet_uint();
  unsigned x = 0;
  __ESBMC_assume(n < 5);
  for (unsigned i = 0; i < n; ++i)
    ++x;
  assert(x == n);
  assert(x < 5);
}
