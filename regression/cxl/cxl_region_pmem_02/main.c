// CXL PMEM driver that sizes its mapping array from the region's target count
// without bounding it.
// Expected: VERIFICATION FAILED (driver bug: out-of-bounds write, CWE-787)

#include <stddef.h>
#include <string.h>
#include <assert.h>
#include <errno.h>
#include <ubuntu20.04/kernel_5.15.0-76/include/linux/cxl.h>

extern unsigned int __VERIFIER_nondet_uint(void);

static u64 mappings[CXL_REGION_MAX_TARGETS];

int main()
{
  struct cxl_region_params p;

  memset(&p, 0, sizeof(p));
  p.state = CXL_CONFIG_COMMIT;
  p.res_start = 0x100000000ULL;
  p.res_end = 0x13FFFFFFFULL;
  p.nr_targets = __VERIFIER_nondet_uint();
  __ESBMC_assume(p.nr_targets >= 1 && p.nr_targets <= 64);

  /*
   * BUG: only one failure is recognised. cxl_pmem_region_alloc() rejects an
   * uncommitted region with -ENXIO and an out-of-range target count with
   * -EINVAL; testing for the first and treating everything else as success
   * turns the second rejection into a go-ahead, and the count that was just
   * refused is then used to index the mapping table.
   */
  if (cxl_pmem_region_alloc(&p) == -ENXIO)
    return 0;

  for (unsigned int i = 0; i < p.nr_targets; i++)
    mappings[i] = p.res_start + i;

  return 0;
}
