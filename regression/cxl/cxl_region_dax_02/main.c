// CXL DAX driver that computes a region's length from the range it was handed
// without checking the allocation succeeded.
// Expected: VERIFICATION FAILED (driver bug: range used on the error path,
// where it was never written)

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
  /* Deliberately the wrong way round, so an unwritten pair is detectable. */
  u64 start = HPA_BASE + HPA_SIZE;
  u64 end = HPA_BASE;

  memset(&p, 0, sizeof(p));
  p.res_start = HPA_BASE;
  p.res_end = HPA_BASE + HPA_SIZE - 1;
  p.nr_targets = 2;
  p.state = (enum cxl_config_state)(__VERIFIER_nondet_uint() % 5);

  /*
   * BUG: the return code is dropped. On -ENXIO the range is left exactly as
   * the caller had it, and a region that is merely ACTIVE rather than
   * COMMIT looks fully configured from everywhere except this return value.
   */
  cxl_dax_region_alloc(&p, &start, &end);

  assert(cxl_range_len(start, end) == HPA_SIZE);
  return 0;
}
