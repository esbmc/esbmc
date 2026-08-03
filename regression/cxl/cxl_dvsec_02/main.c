// CXL driver that reads a DVSEC range's base and size without waiting for
// the range to report MEM_INFO_VALID.
// Expected: VERIFICATION FAILED (driver bug: range consumed before the device
// published it)

#include <stddef.h>
#include <string.h>
#include <assert.h>
#include <errno.h>
#include <ubuntu20.04/kernel_5.15.0-76/include/linux/pci.h>
#include <ubuntu20.04/kernel_5.15.0-76/include/linux/cxl.h>

#define CXL_DVSEC_OFFSET 0x100

struct cxl_dvsec_range
{
  u64 base;
  u64 size;
  int valid;
};

int main()
{
  struct pci_dev dev;
  struct cxl_dvsec_range r;
  u32 size_lo, size_hi, base_lo, base_hi;

  memset(&dev, 0, sizeof(dev));
  memset(&r, 0, sizeof(r));

  /*
   * BUG: MEM_INFO_VALID is never consulted. The real driver polls it for up
   * to a second (cxl_dvsec_mem_range_valid) because the device publishes the
   * range asynchronously -- the registers are readable long before they mean
   * anything. Reading them is not the same as the device having filled them.
   */
  if (pci_read_config_dword(&dev, CXL_DVSEC_OFFSET +
                                    PCI_DVSEC_CXL_RANGE_SIZE_LOW(0), &size_lo))
    return 0;
  if (pci_read_config_dword(&dev, CXL_DVSEC_OFFSET +
                                    PCI_DVSEC_CXL_RANGE_SIZE_HIGH(0), &size_hi))
    return 0;
  if (pci_read_config_dword(&dev, CXL_DVSEC_OFFSET +
                                    PCI_DVSEC_CXL_RANGE_BASE_LOW(0), &base_lo))
    return 0;
  if (pci_read_config_dword(&dev, CXL_DVSEC_OFFSET +
                                    PCI_DVSEC_CXL_RANGE_BASE_HIGH(0), &base_hi))
    return 0;

  r.size = ((u64)size_hi << 32) | (size_lo & 0xF0000000U);
  r.base = ((u64)base_hi << 32) | (base_lo & 0xF0000000U);
  r.valid = 1;

  /* Having decoded a range, the driver believes the device published it. */
  assert(cxl_dvsec_mem_range_valid(&dev, CXL_DVSEC_OFFSET, 0) == 1);

  return 0;
}
