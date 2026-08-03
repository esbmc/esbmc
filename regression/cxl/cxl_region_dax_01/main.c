// CXL DAX region allocation, gated on the region config state machine.
//
// cxl_dax_region_alloc() refuses anything but CXL_CONFIG_COMMIT: a region
// only reaches COMMIT once its geometry is fixed and its targets attached, so
// the gate is what stops a half-configured region reaching DAX.
// Expected: VERIFICATION SUCCESSFUL

#include <stddef.h>
#include <string.h>
#include <assert.h>
#include <errno.h>
#include <ubuntu20.04/kernel_5.15.0-76/include/linux/cxl.h>

extern unsigned int __VERIFIER_nondet_uint(void);

#define HPA_BASE 0x100000000ULL
#define HPA_SIZE 0x40000000ULL

int main()
{
  struct cxl_region_params p;
  u64 start, end;

  memset(&p, 0, sizeof(p));
  p.res_start = HPA_BASE;
  p.res_end = HPA_BASE + HPA_SIZE - 1;
  p.nr_targets = 2;

  /* Every state short of COMMIT is refused, and refusal is the same -ENXIO
     the driver returns. */
  enum cxl_config_state s = (enum cxl_config_state)(__VERIFIER_nondet_uint() % 5);
  p.state = s;
  int rc = cxl_dax_region_alloc(&p, &start, &end);
  if (s == CXL_CONFIG_COMMIT)
  {
    assert(rc == 0);
    assert(start == HPA_BASE);
    assert(end == HPA_BASE + HPA_SIZE - 1);
  }
  else
    assert(rc == -ENXIO);

  /* The range is inclusive at both ends, so its length is end - start + 1. */
  p.state = CXL_CONFIG_COMMIT;
  assert(cxl_dax_region_alloc(&p, &start, &end) == 0);
  assert(cxl_range_len(start, end) == HPA_SIZE);

  /* A single-byte region is representable; an empty one is not. */
  p.res_end = p.res_start;
  assert(cxl_dax_region_alloc(&p, &start, &end) == 0);
  assert(cxl_range_len(start, end) == 1);

  return 0;
}
