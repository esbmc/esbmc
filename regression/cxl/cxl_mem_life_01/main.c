// CXL memory device lifecycle: attach, enable, query regions and partition,
// then unwind in reverse.
//
// cxl_mem_attach() allocates, and the partition sizes come back from the
// device unconstrained -- summing them is the driver's arithmetic, not the
// device's promise.
// Expected: VERIFICATION SUCCESSFUL

#include <stddef.h>
#include <string.h>
#include <assert.h>
#include <errno.h>
#include <ubuntu20.04/kernel_5.15.0-76/include/linux/pci.h>
#include <ubuntu20.04/kernel_5.15.0-76/include/linux/cxl.h>
#include <ubuntu20.04/kernel_5.15.0-76/include/linux/cxlmem.h>

#define CXL_REGIONS_MAX 4

int main()
{
  struct pci_dev pdev;
  struct cxl_dev cxld;
  struct cxl_mem *cxlmem;
  struct cxl_memregion_info regions[CXL_REGIONS_MAX];
  u32 data_size, pmem_size;

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
  assert(cxlmem->cxld == &cxld);

  if (cxl_mem_enable(cxlmem) == 0)
  {
    int n = cxl_mem_get_regions(cxlmem, regions, CXL_REGIONS_MAX);
    assert(n >= 1 && n <= CXL_REGIONS_MAX);
    for (int i = 0; i < n; i++)
      assert(regions[i].size > 0);

    if (cxl_mem_get_partition_state(cxlmem, &data_size, &pmem_size) == 0)
    {
      /* The two halves are reported independently, so their sum is the
         driver's problem: widen rather than trust it to fit. */
      u64 total = (u64)data_size + (u64)pmem_size;
      assert(total >= data_size);
    }

    if (pmem_size > 0)
      (void)cxl_mem_set_pmem_capacity(cxlmem, pmem_size);
    (void)cxl_mem_set_partition_state(cxlmem, data_size, pmem_size);

    cxl_mem_flush(cxlmem);
    cxl_mem_disable(cxlmem);
  }

  cxl_mem_detach(cxlmem);
  pci_iounmap(&pdev, cxld.regs);
  return 0;
}
