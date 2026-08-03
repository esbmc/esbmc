// CXL RAS handler that uses the reported first error without checking an
// error was reported at all.
// Expected: VERIFICATION FAILED (driver bug: fatal error decoded from an
// unwritten first-error word)

#include <stddef.h>
#include <assert.h>
#include <errno.h>
#include <ubuntu20.04/kernel_5.15.0-76/include/linux/cxl.h>

extern unsigned int __VERIFIER_nondet_uint(void);

#define CXL_RAS_UC_FATAL_BIT 0x4U

int main()
{
  u32 status = __VERIFIER_nondet_uint();
  u32 fe = CXL_RAS_UC_FATAL_BIT; /* stale value from a previous interrupt */

  /*
   * BUG: the return value is dropped. cxl_handle_ras() leaves *fe untouched
   * when nothing was latched, so a spurious interrupt -- or one raised for a
   * correctable error -- is decoded against whatever the last uncorrectable
   * error left behind, and the driver escalates a fatal that is not there.
   */
  cxl_handle_ras(status, 0, &fe);

  if (fe & CXL_RAS_UC_FATAL_BIT)
    assert(status & CXL_RAS_UNCORRECTABLE_STATUS_MASK);

  return 0;
}
