// CXL driver that programs HDM decoders after asking for an unlock, without
// checking the unlock happened.
// Expected: VERIFICATION FAILED (driver bug: device still locked when its
// decoders are programmed)

#include <stddef.h>
#include <string.h>
#include <assert.h>
#include <errno.h>
#include <ubuntu20.04/kernel_5.15.0-76/include/linux/pci.h>
#include <ubuntu20.04/kernel_5.15.0-76/include/linux/cxl.h>

#define CXL_REGION_BASE 0x100000000ULL
#define CXL_REGION_SIZE 0x40000000ULL

int main()
{
  struct pci_dev pdev;
  struct cxl_dev cxld;
  struct cxl_region region;

  memset(&pdev, 0, sizeof(pdev));
  memset(&cxld, 0, sizeof(cxld));
  cxld.regs = pci_iomap(&pdev, 0, 4096);
  if (!cxld.regs)
    return 0;
  cxld.pdev = &pdev;

  enum cxl_security_state s = cxl_get_security_state(&cxld);

  /*
   * BUG: the transition is requested and the result dropped. cxl_set_security()
   * can be refused -- a device with a passphrase set will refuse -- and the
   * driver then programs the decoders of a device that is still locked. The
   * request is not the state.
   */
  if (s == CXL_SEC_LOCKED)
    cxl_set_security(&cxld, CXL_SEC_UNLOCKED);

  assert(cxl_get_security_state(&cxld) != CXL_SEC_LOCKED);

  region.start = CXL_REGION_BASE;
  region.size = CXL_REGION_SIZE;
  region.granularity = 256;
  region.ways = 1;
  (void)cxl_setup_hdm_decoders(&cxld, &region);

  pci_iounmap(&pdev, cxld.regs);
  return 0;
}
