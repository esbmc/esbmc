// CXL driver that keeps the device cxl_find_device() handed it after
// releasing the bridge it came from.
// Expected: VERIFICATION FAILED (driver bug: use-after-free, CWE-416)

#include <stddef.h>
#include <string.h>
#include <assert.h>
#include <ubuntu20.04/kernel_5.15.0-76/include/linux/pci.h>
#include <ubuntu20.04/kernel_5.15.0-76/include/linux/cxl.h>

#define CXL_VENDOR 0x8086
#define CXL_DEVICE 0x0d93

int main()
{
  struct cxl_host_bridge *bridge;
  struct cxl_dev *cxld;

  esbmc_pci_reset_devices();

  bridge = cxl_enumerate_ports();
  if (!bridge)
    return 0;

  cxld = cxl_find_device(bridge, CXL_VENDOR, CXL_DEVICE);
  if (!cxld)
  {
    cxl_free_ports(bridge);
    return 0;
  }

  /*
   * BUG: cxl_find_device() does not hand out a reference, it hands out a
   * pointer into the bridge's own device array. Freeing the bridge frees
   * that array, and a device handle is not the sort of thing that looks
   * stale afterwards -- it still points somewhere.
   */
  cxl_free_ports(bridge);

  assert(cxld->regs != NULL);
  return 0;
}
