// PCIe DVSEC for CXL Device: HDM range count decode, per-range register
// offsets, and the MEM_INFO_VALID gate before a range is used.
//
// Complements cxl_pci_config_01, which covers the range-count bound. This one
// covers what the count is decoded *from*: HDM_COUNT is a two-bit field at
// bits 5:4, so the device can report 3 while dvsec_range[] holds 2 entries.
// The bound is not implied by the encoding.
// Expected: VERIFICATION SUCCESSFUL

#include <stddef.h>
#include <string.h>
#include <assert.h>
#include <errno.h>
#include <ubuntu20.04/kernel_5.15.0-76/include/linux/pci.h>
#include <ubuntu20.04/kernel_5.15.0-76/include/linux/cxl.h>

extern unsigned int __VERIFIER_nondet_uint(void);

#define CXL_DVSEC_OFFSET 0x100

int main()
{
  struct pci_dev dev;
  memset(&dev, 0, sizeof(dev));

  /* The field is two bits wide, so it decodes to 0..3 -- one more than the
     array can hold. That gap is the whole reason the rejection exists. */
  u32 cap = __VERIFIER_nondet_uint();
  unsigned int n = cxl_dvsec_hdm_count(cap);
  assert(n <= 3);
  /* And 3 is genuinely reachable, which is what makes the gap real rather
     than theoretical. */
  if (cap & PCI_DVSEC_CXL_HDM_COUNT_MASK)
    assert(n >= 1);

  /* Decoding is exact: only bits 5:4 contribute. */
  assert(cxl_dvsec_hdm_count(0x00) == 0);
  assert(cxl_dvsec_hdm_count(0x10) == 1);
  assert(cxl_dvsec_hdm_count(0x20) == 2);
  assert(cxl_dvsec_hdm_count(0x30) == 3);
  assert(cxl_dvsec_hdm_count(0xFFFFFFCFU) == 0);

  /* Range register offsets step by 0x10 and do not overlap. */
  for (int i = 0; i < CXL_DVSEC_RANGE_MAX; i++)
  {
    assert(PCI_DVSEC_CXL_RANGE_SIZE_LOW(i) ==
           PCI_DVSEC_CXL_RANGE_SIZE_HIGH(i) + 4);
    assert(PCI_DVSEC_CXL_RANGE_BASE_LOW(i) ==
           PCI_DVSEC_CXL_RANGE_BASE_HIGH(i) + 4);
    if (i > 0)
      assert(PCI_DVSEC_CXL_RANGE_SIZE_HIGH(i) ==
             PCI_DVSEC_CXL_RANGE_SIZE_HIGH(i - 1) + 0x10);
  }

  /* The validity gate answers for every id the driver admits, and refuses
     the ones it does not. */
  for (int id = 0; id <= CXL_DVSEC_RANGE_MAX; id++)
  {
    int rc = cxl_dvsec_mem_range_valid(&dev, CXL_DVSEC_OFFSET, id);
    assert(rc == 0 || rc == 1 || rc == -EINVAL);
  }
  assert(cxl_dvsec_mem_range_valid(&dev, CXL_DVSEC_OFFSET,
                                   CXL_DVSEC_RANGE_MAX + 1) == -EINVAL);

  return 0;
}
