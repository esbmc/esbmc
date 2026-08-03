// CXL RAS uncorrectable error handling (drivers/cxl/core/ras.c).
//
// cxl_handle_ras() reports whether an uncorrectable error was latched, and
// names the first error. With several bits set the header log describes the
// one the capability control register points at, not the whole status word --
// so "which error" and "were there errors" are different questions.
// Expected: VERIFICATION SUCCESSFUL

#include <stddef.h>
#include <assert.h>
#include <errno.h>
#include <ubuntu20.04/kernel_5.15.0-76/include/linux/cxl.h>

extern unsigned int __VERIFIER_nondet_uint(void);

int main()
{
  u32 fe;

  /* Nothing latched: nothing handled, and fe is not touched. */
  assert(cxl_handle_ras(0, 0, &fe) == 0);
  assert(cxl_handle_ras(~CXL_RAS_UNCORRECTABLE_STATUS_MASK, 0, &fe) == 0);

  /* Exactly one error: the first error is that error. */
  fe = 0;
  assert(cxl_handle_ras(0x1, 0, &fe) == 1);
  assert(fe == 0x1);

  /* Several errors: the control register selects which, and the answer is a
     single bit whichever way it was reached. */
  u32 status = __VERIFIER_nondet_uint();
  u32 idx = __VERIFIER_nondet_uint() % 32;
  fe = 0;
  if (cxl_handle_ras(status, idx, &fe) == 1)
  {
    assert(status & CXL_RAS_UNCORRECTABLE_STATUS_MASK);
    assert(fe != 0);
    /* A first error is one error. */
    if (__builtin_popcount(status & CXL_RAS_UNCORRECTABLE_STATUS_MASK) > 1)
      assert((fe & (fe - 1)) == 0);
  }
  else
    assert(!(status & CXL_RAS_UNCORRECTABLE_STATUS_MASK));

  return 0;
}
