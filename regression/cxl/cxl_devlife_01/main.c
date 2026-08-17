// CXL device bring-up and teardown through the real register path: map the
// BAR, register the driver, init, then unwind.
//
// cxl_device_init() drives DEV_CTRL and reports whether the device came up;
// its failure is nondeterministic, so the caller has to unwind either way.
// Expected: VERIFICATION SUCCESSFUL

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

  /* Register state must come from the mapping, not from a buffer the harness
     invented: the model reads and writes it as device memory. */
  cxld.regs = pci_iomap(&pdev, CXL_REGS_BAR, CXL_REGS_LEN);
  if (!cxld.regs)
  {
    cxl_driver_unregister(&drv);
    return 0;
  }
  cxld.pdev = &pdev;

  if (cxl_device_init(&cxld) == 0)
  {
    /* Init reports success only after setting ENABLE and clearing INIT. */
    u64 ctrl = cxl_read_dev_ctrl(&cxld);
    assert(ctrl & CXL_DCR_ENABLE);
    assert(!(ctrl & CXL_DCR_CLEAR_INIT));

    /* DEV_STAT is a separate register; reading it must not disturb DEV_CTRL. */
    (void)cxl_read_dev_stat(&cxld);
    assert(cxl_read_dev_ctrl(&cxld) == ctrl);

    /* A driver may drive DEV_CTRL directly to request a reset. The register
       is 64-bit, so a read-modify-write has to preserve the upper half. */
    cxl_write_dev_ctrl(&cxld, ctrl | CXL_DCR_RESET);
    assert(cxl_read_dev_ctrl(&cxld) == (ctrl | CXL_DCR_RESET));
    cxl_write_dev_ctrl(&cxld, ctrl);

    cxl_device_exit(&cxld);
    assert(cxl_read_dev_ctrl(&cxld) == 0);
  }

  pci_iounmap(&pdev, cxld.regs);
  cxl_driver_unregister(&drv);
  return 0;
}
