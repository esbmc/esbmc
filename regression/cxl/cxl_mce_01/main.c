// CXL MCE notifier: offlining the page that aliases a poisoned SPA
// (drivers/cxl/core/mce.c).
//
// The original page is the standard MCE handler's job; this notifier exists
// only for the alias, and not every SPA has one.
// Expected: VERIFICATION SUCCESSFUL

#include <stddef.h>
#include <assert.h>
#include <errno.h>
#include <ubuntu20.04/kernel_5.15.0-76/include/linux/cxl.h>

extern unsigned long __VERIFIER_nondet_ulong(void);

int main()
{
  u64 pfn = 0;

  /* No alias: nothing to take down, and the caller's pfn is untouched. */
  assert(cxl_mce_offline_page(CXL_SPA_NO_ALIAS, &pfn) == 0);
  assert(pfn == 0);

  /* An alias yields the page frame containing it. */
  u64 alias = 0x200000000ULL;
  assert(cxl_mce_offline_page(alias, &pfn) == 1);
  assert(pfn == alias >> CXL_PAGE_SHIFT);

  /* Any address inside the same page maps to the same frame -- the offset
     within the page is not part of the answer. */
  u64 nd = (u64)__VERIFIER_nondet_ulong();
  __ESBMC_assume(nd != CXL_SPA_NO_ALIAS);
  u64 pfn2 = 0;
  if (cxl_mce_offline_page(nd, &pfn2) == 1)
  {
    assert(pfn2 == nd >> CXL_PAGE_SHIFT);
    assert(pfn2 <= nd);
  }

  return 0;
}
