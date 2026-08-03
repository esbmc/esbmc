// CXL PMEM region allocation: the same commit gate as DAX, plus the target
// count that sizes its flexible array.
//
// cxl_pmem_region_alloc() does kzalloc_flex(*cxlr_pmem, mapping,
// p->nr_targets) -- the array is sized by a device-influenced count, so the
// count has to be bounded before it is used.
// Expected: VERIFICATION SUCCESSFUL

#include <stddef.h>
#include <string.h>
#include <assert.h>
#include <errno.h>
#include <ubuntu20.04/kernel_5.15.0-76/include/linux/cxl.h>

extern unsigned int __VERIFIER_nondet_uint(void);

int main()
{
  struct cxl_region_params p;

  memset(&p, 0, sizeof(p));
  p.res_start = 0x100000000ULL;
  p.res_end = 0x13FFFFFFFULL;

  /* Uncommitted regions are refused whatever their target count. */
  p.nr_targets = 4;
  for (unsigned int s = 0; s < CXL_CONFIG_COMMIT; s++)
  {
    p.state = (enum cxl_config_state)s;
    assert(cxl_pmem_region_alloc(&p) == -ENXIO);
  }

  /* Committed and in range: accepted. */
  p.state = CXL_CONFIG_COMMIT;
  assert(cxl_pmem_region_alloc(&p) == 0);

  /* Acceptance implies the count is one the flexible array can hold. */
  p.nr_targets = __VERIFIER_nondet_uint();
  if (cxl_pmem_region_alloc(&p) == 0)
    assert(p.nr_targets >= 1 && p.nr_targets <= CXL_REGION_MAX_TARGETS);

  /* A region with no targets is not a region. */
  p.nr_targets = 0;
  assert(cxl_pmem_region_alloc(&p) == -EINVAL);

  return 0;
}
