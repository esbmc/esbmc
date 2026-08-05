// CXL streaming DMA: a kmalloc'd descriptor buffer mapped for the device,
// synced back, unmapped, then freed.
//
// Unlike the coherent case (cxl_dma_coherent_01), a streaming mapping hands
// the buffer to the device for the whole map/unmap window. The CPU may not
// touch it in between without a sync, and the buffer must outlive the
// mapping.
// Expected: VERIFICATION SUCCESSFUL

#include <stddef.h>
#include <stdint.h>
#include <assert.h>
#include <ubuntu20.04/kernel_5.15.0-76/include/linux/dma-mapping.h>
#include <ubuntu20.04/kernel_5.15.0-76/include/linux/slab.h>
#include <ubuntu20.04/kernel_5.15.0-76/include/linux/gfp.h>

#define CXL_DESC_LEN 64
#define CXL_DMA_MASK 0xFFFFFFFFFFFFULL /* 48-bit, per CXL.mem HPA width */

int main()
{
  struct device dev;
  uint8_t *desc;
  dma_addr_t handle;

  /* A driver that ignores these and then maps is claiming an addressing
     capability the platform never granted. */
  if (dma_set_mask(&dev, CXL_DMA_MASK))
    return 0;
  if (dma_set_coherent_mask(&dev, CXL_DMA_MASK))
    return 0;

  desc = kmalloc(CXL_DESC_LEN, GFP_KERNEL);
  if (!desc)
    return 0;

  desc[0] = 0xC9;

  handle = dma_map_single(&dev, desc, CXL_DESC_LEN, DMA_BIDIRECTIONAL);

  /* Ownership is the device's here. Reclaim it before reading. */
  dma_sync_single_for_cpu(&dev, handle, CXL_DESC_LEN, DMA_FROM_DEVICE);
  desc[1] = desc[0];
  dma_sync_single_for_device(&dev, handle, CXL_DESC_LEN, DMA_TO_DEVICE);

  dma_unmap_single(&dev, handle, CXL_DESC_LEN, DMA_BIDIRECTIONAL);

  /* Only now is the buffer the CPU's alone again. */
  assert(desc[1] == 0xC9);
  kfree(desc);
  return 0;
}
