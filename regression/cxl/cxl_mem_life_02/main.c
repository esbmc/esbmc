// CXL memory device teardown that detaches before quiescing the device.
// Expected: VERIFICATION FAILED (driver bug: use-after-free, CWE-416)

#include <stddef.h>
#include <string.h>
#include <assert.h>
#include <errno.h>
#include <ubuntu20.04/kernel_5.15.0-76/include/linux/pci.h>
#include <ubuntu20.04/kernel_5.15.0-76/include/linux/cxl.h>
#include <ubuntu20.04/kernel_5.15.0-76/include/linux/cxlmem.h>

int main()
{
  struct pci_dev pdev;
  struct cxl_dev cxld;
  struct cxl_mem *cxlmem;

  memset(&pdev, 0, sizeof(pdev));
  memset(&cxld, 0, sizeof(cxld));
  cxld.regs = pci_iomap(&pdev, 0, 4096);
  if (!cxld.regs)
    return 0;
  cxld.pdev = &pdev;

  cxlmem = cxl_mem_attach(&cxld);
  if (!cxlmem)
  {
    pci_iounmap(&pdev, cxld.regs);
    return 0;
  }

  if (cxl_mem_enable(cxlmem) == 0)
  {
    /*
     * BUG: detach releases the cxl_mem, and the remaining teardown still
     * needs it -- cxl_mem_disable() reaches through it to the device's
     * registers. Releasing the handle is the last step of teardown, not the
     * first; the device is still running until it is told otherwise.
     */
    cxl_mem_detach(cxlmem);

    cxl_mem_flush(cxlmem);
    cxl_mem_disable(cxlmem);
  }
  else
  {
    cxl_mem_detach(cxlmem);
  }

  pci_iounmap(&pdev, cxld.regs);
  return 0;
}
