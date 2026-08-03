// CXL AER handler that indexes its per-severity counters with a severity it
// never confirmed was written.
// Expected: VERIFICATION FAILED (driver bug: out-of-bounds store, CWE-787)

#include <stddef.h>
#include <assert.h>
#include <errno.h>
#include <string.h>
#include <ubuntu20.04/kernel_5.15.0-76/include/linux/pci.h>
#include <ubuntu20.04/kernel_5.15.0-76/include/linux/cxl.h>

#define CXL_AER_SEVERITIES 3

static unsigned long cxl_aer_stats[CXL_AER_SEVERITIES];

int main()
{
  struct pci_dev dev;
  int severity;

  memset(&dev, 0, sizeof(dev));
  dev.vendor = 0x8086;
  dev.device = 0x0d93;

  /*
   * BUG: the return code is dropped. pci_aer_get_first_error() leaves
   * *severity untouched when reporting is not enabled -- and nothing here
   * enabled it -- so severity is whatever was on the stack. The read looks
   * total because the register always exists; what does not always exist is
   * a value having been written to it.
   */
  pci_aer_get_first_error(&dev, &severity);
  cxl_aer_stats[severity]++;

  assert(cxl_aer_stats[severity] > 0);
  return 0;
}
