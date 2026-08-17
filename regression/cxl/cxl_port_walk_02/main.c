// CXL host-bridge probe that reads the first downstream port without first
// asking whether there is one.
// Expected: VERIFICATION FAILED (driver bug: NULL dereference, CWE-476)

#include <stddef.h>
#include <assert.h>
#include <ubuntu20.04/kernel_5.15.0-76/include/linux/cxl.h>

int main()
{
  struct cxl_host_bridge *bridge;

  bridge = cxl_enumerate_ports();
  if (!bridge)
    return 0;

  /*
   * BUG: a bridge that enumerated no downstream ports carries no device
   * array. Checking the bridge pointer is not the same as checking the
   * array, and "a host bridge always has at least one endpoint below it" is
   * an assumption about the platform, not something the enumeration
   * promised.
   */
  struct cxl_dev *first = &bridge->devices[0];
  assert(first->regs != NULL);

  cxl_free_ports(bridge);
  return 0;
}
