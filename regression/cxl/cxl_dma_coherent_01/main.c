// CXL DMA-coherent buffer with ownership handed over explicitly: the CPU
// fills the buffer, releases it to the device, and only then does the device
// side touch it.
// Expected: VERIFICATION SUCCESSFUL

#include <pthread.h>
#include <stdint.h>
#include <stddef.h>
#include <assert.h>
#include <ubuntu20.04/kernel_5.15.0-76/include/linux/gfp.h>
#include <ubuntu20.04/kernel_5.15.0-76/include/linux/dma-mapping.h>

#define DMA_LEN 64

static uint32_t *shared;

/* Guards ownership of the buffer, not its contents: whoever holds it may
   touch the buffer, and no one else may. */
static pthread_mutex_t owner = PTHREAD_MUTEX_INITIALIZER;

static void *device_side(void *arg)
{
  (void)arg;

  pthread_mutex_lock(&owner);
  /* The device observes what the CPU wrote — that is what coherent means —
     and the handover is what makes the observation well defined. */
  assert(shared[0] == 0xC9U);
  shared[0] = 0xD1CEU;
  pthread_mutex_unlock(&owner);

  return NULL;
}

int main()
{
  struct device dev;
  dma_addr_t handle;

  shared = (uint32_t *)dma_alloc_coherent(&dev, DMA_LEN, &handle, GFP_KERNEL);
  if (shared == NULL)
    return 0;

  pthread_mutex_lock(&owner);
  shared[0] = 0xC9U;
  pthread_mutex_unlock(&owner);

  pthread_t d;
  pthread_create(&d, NULL, device_side, NULL);
  pthread_join(d, NULL);

  pthread_mutex_lock(&owner);
  assert(shared[0] == 0xD1CEU);
  pthread_mutex_unlock(&owner);

  dma_free_coherent(&dev, DMA_LEN, shared, handle);
  return 0;
}
