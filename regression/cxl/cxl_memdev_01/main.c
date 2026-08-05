// CXL memdev creation: a driver that checks the id allocator ends up with
// a well-formed /dev/cxl/memN device.
// Expected: VERIFICATION SUCCESSFUL

#include <stdint.h>
#include <stddef.h>
#include <assert.h>
#include <ubuntu20.04/kernel_5.15.0-76/include/linux/cxl.h>
#include <ubuntu20.04/kernel_5.15.0-76/include/linux/cxlmem.h>

int main()
{
  struct pci_dev pdev;
  struct cxl_dev cxld;
  cxld.pdev = &pdev;
  cxld.dev_type = CXL_TYPE_RAM;

  struct cxl_memdev *cxlmd = cxl_memdev_create(&cxld);

  /* Creation is fallible — the id space may be exhausted, or the
     allocation may fail — so a correct driver tolerates NULL. */
  if (cxlmd == NULL)
    return 0;

  /* The minor number names /dev/cxl/memN, so it must be a usable index. */
  assert(cxlmd->id >= 0);
  assert(cxlmd->id < CXL_MEM_MAX_DEVS);

  /* The revision string is opaque, but always terminated. */
  assert(cxlmd->fw_rev[CXL_MEMDEV_FW_REV_LEN - 1] == '\0');

  assert(cxlmd->live != 0);
  assert(cxlmd->cxld == &cxld);

  cxl_memdev_destroy(cxlmd);
  return 0;
}
