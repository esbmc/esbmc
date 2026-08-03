// CXL streaming DMA whose error path frees the descriptor buffer while it is
// still mapped, then completes the teardown through the mapping.
// Expected: VERIFICATION FAILED (driver bug: use-after-free, CWE-416)

#include <stddef.h>
#include <stdint.h>
#include <assert.h>
#include <ubuntu20.04/kernel_5.15.0-76/include/linux/dma-mapping.h>
#include <ubuntu20.04/kernel_5.15.0-76/include/linux/slab.h>
#include <ubuntu20.04/kernel_5.15.0-76/include/linux/gfp.h>

#define CXL_DESC_LEN 64
#define CXL_DMA_MASK 0xFFFFFFFFFFFFULL

int main()
{
  struct device dev;
  uint8_t *desc;
  dma_addr_t handle;

  if (dma_set_mask(&dev, CXL_DMA_MASK))
    return 0;

  desc = kmalloc(CXL_DESC_LEN, GFP_KERNEL);
  if (!desc)
    return 0;

  desc[0] = 0xC9;
  handle = dma_map_single(&dev, desc, CXL_DESC_LEN, DMA_BIDIRECTIONAL);

  /*
   * BUG: the buffer is released while the mapping still refers to it. kfree()
   * looks like the right first step of an unwind because it undoes the first
   * acquisition -- but teardown has to run in reverse order, and the mapping
   * outlives the allocation here.
   */
  kfree(desc);

  dma_sync_single_for_cpu(&dev, handle, CXL_DESC_LEN, DMA_FROM_DEVICE);
  assert(desc[0] == 0xC9);
  dma_unmap_single(&dev, handle, CXL_DESC_LEN, DMA_BIDIRECTIONAL);
  return 0;
}
