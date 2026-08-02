// CXL DMA-coherent buffer accessed by CPU and device concurrently, with
// nothing serialising them.
// Expected: VERIFICATION FAILED (driver bug: unsynchronised DMA buffer access)

#include <pthread.h>
#include <stdint.h>
#include <stddef.h>
#include <assert.h>
#include <ubuntu20.04/kernel_5.15.0-76/include/linux/gfp.h>
#include <ubuntu20.04/kernel_5.15.0-76/include/linux/dma-mapping.h>

#define DMA_LEN 64

static uint32_t *shared;

/*
 * BUG: "coherent" means the CPU and device see one another's writes without
 * explicit cache maintenance — it does not mean the two may write the same
 * word at the same time. Ownership of the buffer still has to be handed over.
 * This driver hands nothing over.
 */
static void *device_side(void *arg)
{
  (void)arg;
  shared[0] = 0xD1CE;
  return NULL;
}

int main()
{
  struct device dev;
  dma_addr_t handle;

  shared = (uint32_t *)dma_alloc_coherent(&dev, DMA_LEN, &handle, GFP_KERNEL);
  if (shared == NULL)
    return 0;

  pthread_t d;
  pthread_create(&d, NULL, device_side, NULL);

  shared[0] = 0xC9U;

  pthread_join(d, NULL);

  dma_free_coherent(&dev, DMA_LEN, shared, handle);
  return 0;
}
