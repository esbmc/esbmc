// CXL security state transitions and HDM decoder setup.
//
// cxl_get_security_state() reports whatever the device is in, which is always
// one of the five defined states -- a driver switching on it must still cope
// with every one, including the ones its own transitions never produce.
// Expected: VERIFICATION SUCCESSFUL

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
  assert(s >= CXL_SEC_NONE && s <= CXL_SEC_PASSPHRASE_SET);

  /* A locked device must be unlocked before its decoders can be programmed;
     asking for the transition is not the same as getting it. */
  if (s == CXL_SEC_LOCKED)
  {
    if (cxl_set_security(&cxld, CXL_SEC_UNLOCKED))
    {
      pci_iounmap(&pdev, cxld.regs);
      return 0;
    }
  }

  region.start = CXL_REGION_BASE;
  region.size = CXL_REGION_SIZE;
  region.granularity = 256;
  region.ways = 1;
  assert(region.start % CXL_HDM_ALIGNMENT == 0);

  /* Decoder setup is fallible: the device may already have all 8 in use. */
  (void)cxl_setup_hdm_decoders(&cxld, &region);

  pci_iounmap(&pdev, cxld.regs);
  return 0;
}
