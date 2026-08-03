// CXL bring-up that treats cxl_device_init() as infallible.
// Expected: VERIFICATION FAILED (driver bug: device used without confirming
// it came up)

#include <stddef.h>
#include <string.h>
#include <assert.h>
#include <errno.h>
#include <ubuntu20.04/kernel_5.15.0-76/include/linux/pci.h>
#include <ubuntu20.04/kernel_5.15.0-76/include/linux/cxl.h>

#define CXL_REGS_BAR 0
#define CXL_REGS_LEN 4096

int main()
{
  struct pci_dev pdev;
  struct cxl_driver drv;
  struct cxl_dev cxld;

  memset(&pdev, 0, sizeof(pdev));
  memset(&cxld, 0, sizeof(cxld));

  drv.name = "cxl_core";
  drv.probe = NULL;
  drv.remove = NULL;
  drv.ids = NULL;
  drv.nids = 0;
  if (cxl_driver_register(&drv))
    return 0;

  cxld.regs = pci_iomap(&pdev, CXL_REGS_BAR, CXL_REGS_LEN);
  if (!cxld.regs)
  {
    cxl_driver_unregister(&drv);
    return 0;
  }
  cxld.pdev = &pdev;

  /*
   * BUG: the return code is dropped. Device init is a handshake with
   * hardware -- it writes INIT and waits for the device to come back -- and
   * the device is entitled not to. Having asked it to enable is not the same
   * as it having enabled.
   */
  cxl_device_init(&cxld);
  assert(cxl_read_dev_ctrl(&cxld) & CXL_DCR_ENABLE);

  cxl_device_exit(&cxld);
  pci_iounmap(&pdev, cxld.regs);
  cxl_driver_unregister(&drv);
  return 0;
}
