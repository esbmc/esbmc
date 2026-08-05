// CXL region configuration state machine as a Linux RV automaton.
//
// cxl_region.dot follows enum cxl_config_state (drivers/cxl/cxl.h): a region
// takes its size, then gains targets, and only then may be committed. This is
// the ordering that cxl_dax_region_alloc() and cxl_pmem_region_alloc() rely on
// when they refuse anything that is not CXL_CONFIG_COMMIT.
// Expected: VERIFICATION SUCCESSFUL

#include <assert.h>
#include "rv_cxl_region.h"

extern unsigned int __VERIFIER_nondet_uint(void);
void __ESBMC_assume(_Bool);

int main(void)
{
  rv_cxl_region_reset();

  rv_cxl_region_event(set_size);
  rv_cxl_region_event(add_target);

  /* Further targets are fine while the region is active. */
  unsigned int extra = __VERIFIER_nondet_uint();
  __ESBMC_assume(extra <= 2);
  for (unsigned int i = 0; i < extra; i++)
    rv_cxl_region_event(add_target);

  rv_cxl_region_event(commit);
  assert(rv_cxl_region_cur == cfg_commit);

  rv_cxl_region_event(reset);
  assert(rv_cxl_region_cur == cfg_idle);

  return 0;
}
