// CXL PMU counter-count decode (drivers/perf/cxl_pmu.c).
//
// The counter count comes from a six-bit capability field with one added, so
// it lands in 1..64 -- exactly CXL_PMU_MAX_COUNTERS. Encoding and array agree
// here, which is worth checking precisely because the DVSEC HDM_COUNT field
// (cxl_dvsec_01) is the case where they do not.
// Expected: VERIFICATION SUCCESSFUL

#include <stddef.h>
#include <assert.h>
#include <errno.h>
#include <ubuntu20.04/kernel_5.15.0-76/include/linux/cxl.h>

extern unsigned long __VERIFIER_nondet_ulong(void);

static u64 counters[CXL_PMU_MAX_COUNTERS];

int main()
{
  /* The +1 is what makes an all-ones field mean "64", not "63". */
  assert(cxl_pmu_num_counters(0) == 1);
  assert(cxl_pmu_num_counters(0x3F) == CXL_PMU_MAX_COUNTERS);

  /* Only the low six bits contribute. */
  assert(cxl_pmu_num_counters(0xFFFFFFFFFFFFFFC0ULL) == 1);

  /* Whatever the register holds, the decoded count indexes the array
     safely -- no clamp needed, because the field cannot encode more. */
  u64 cap = (u64)__VERIFIER_nondet_ulong();
  unsigned int n = cxl_pmu_num_counters(cap);
  assert(n >= 1 && n <= CXL_PMU_MAX_COUNTERS);

  for (unsigned int i = 0; i < n; i++)
    counters[i] = i;
  assert(counters[n - 1] == n - 1);

  return 0;
}
