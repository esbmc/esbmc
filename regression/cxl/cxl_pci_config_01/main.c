// CXL DVSEC range decode over PCI config space, with the range-count
// rejection that keeps the store in bounds.
//
// This is the synthetic counterpart to regression/cxl-linux's
// harness_dvsec_rr_decode: cxl_dvsec_rr_decode() in drivers/cxl/core/pci.c
// fills a 2-entry dvsec_range[] and is kept in bounds solely by its
// "hdm_count > 2" rejection. Config space is hardware state, so the count is
// unconstrained until the driver constrains it.
// Expected: VERIFICATION SUCCESSFUL

#include <stddef.h>
#include <assert.h>
#include <errno.h>
#include <ubuntu20.04/kernel_5.15.0-76/include/linux/pci.h>

#define CXL_DVSEC_CAP 0x100
#define CXL_DVSEC_RANGE_MAX 2

struct cxl_range
{
  u64 base;
  u64 size;
};

struct cxl_endpoint_dvsec_info
{
  int ranges;
  struct cxl_range dvsec_range[CXL_DVSEC_RANGE_MAX];
};

static int cxl_dvsec_decode(struct pci_dev *pdev,
                            struct cxl_endpoint_dvsec_info *info)
{
  u16 cap;
  u32 lo, hi;
  u8 hdm_count;

  if (pci_read_config_word(pdev, CXL_DVSEC_CAP, &cap))
    return -ENXIO;
  if (pci_read_config_byte(pdev, CXL_DVSEC_CAP + 2, &hdm_count))
    return -ENXIO;

  /* Load-bearing: hdm_count comes off the wire, and dvsec_range[] holds
     CXL_DVSEC_RANGE_MAX entries. Dropping this rejection is what the _02
     variant does. */
  if (hdm_count > CXL_DVSEC_RANGE_MAX)
    return -EINVAL;

  info->ranges = 0;
  for (u8 i = 0; i < hdm_count; i++)
  {
    if (pci_read_config_dword(pdev, CXL_DVSEC_CAP + 8 + i * 8, &lo))
      return -ENXIO;
    if (pci_read_config_dword(pdev, CXL_DVSEC_CAP + 12 + i * 8, &hi))
      return -ENXIO;

    info->dvsec_range[i].base = ((u64)hi << 32) | (lo & 0xF0000000U);
    info->dvsec_range[i].size = (u64)(lo & 0x0FFFFFF0U);
    info->ranges++;
  }

  /* Acknowledge the decode by writing the control register back. */
  pci_write_config_word(pdev, CXL_DVSEC_CAP + 4, cap);
  pci_write_config_dword(pdev, CXL_DVSEC_CAP + 16, (u32)info->ranges);
  pci_write_config_byte(pdev, CXL_DVSEC_CAP + 3, hdm_count);

  return 0;
}

int main()
{
  struct pci_dev dev;
  struct cxl_endpoint_dvsec_info info;

  dev.vendor = 0x8086;
  dev.device = 0x0d93;

  if (cxl_dvsec_decode(&dev, &info) == 0)
    assert(info.ranges <= CXL_DVSEC_RANGE_MAX);

  return 0;
}
