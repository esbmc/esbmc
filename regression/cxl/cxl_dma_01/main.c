// CXL DMA access verification test.
// Tests that DMA buffers are properly synced before CPU access
// (dma_sync_single_for_cpu must be called before reading).
// Expected: VERIFICATION FAILED (driver bug: reads DMA buffer without sync)

#include <stdint.h>
#include <string.h>
#include <assert.h>

/* DMA direction */
enum dma_data_direction {
  DMA_BIDIRECTIONAL = 0,
  DMA_TO_DEVICE     = 1,
  DMA_FROM_DEVICE   = 2,
  DMA_NONE          = 3,
};

/* DMA address type */
typedef uint64_t dma_addr_t;

/* Device structure (minimal) */
struct device {
  const char *init_name;
};

/* Simulated device and DMA buffer */
static struct device test_dev;
static uint8_t cpu_buffer[64];
static uint8_t dma_buffer[64];
static int dma_sync_called = 0;

/* Override dma_alloc_coherent for deterministic testing */
void *dma_alloc_coherent(struct device *dev, size_t size,
                         dma_addr_t *dma_handle, unsigned int flag)
{
  (void)dev; (void)flag;
  *dma_handle = 0;
  return dma_buffer;
}

void dma_free_coherent(struct device *dev, size_t size,
                       void *cpu_addr, dma_addr_t dma_handle)
{
  (void)dev; (void)size; (void)cpu_addr; (void)dma_handle;
}

dma_addr_t dma_map_single(struct device *dev, void *cpu_addr, size_t size,
                          enum dma_data_direction dir)
{
  (void)dev; (void)cpu_addr; (void)size; (void)dir;
  return 0;
}

void dma_unmap_single(struct device *dev, dma_addr_t dma_handle, size_t size,
                      enum dma_data_direction dir)
{
  (void)dev; (void)dma_handle; (void)size; (void)dir;
}

void dma_sync_single_for_cpu(struct device *dev, dma_addr_t dma_handle,
                             size_t size, enum dma_data_direction dir)
{
  (void)dev; (void)dma_handle; (void)size; (void)dir;
  dma_sync_called = 1;
  /* In the model, this copies device-written data into CPU buffer */
  memcpy(cpu_buffer, dma_buffer, size);
}

void dma_sync_single_for_device(struct device *dev, dma_addr_t dma_handle,
                                size_t size, enum dma_data_direction dir)
{
  (void)dev; (void)dma_handle; (void)size; (void)dir;
  /* In the model, this copies CPU data into device-visible buffer */
  memcpy(dma_buffer, cpu_buffer, size);
}

/*
 * BUG: This driver reads from a DMA buffer without calling
 * dma_sync_single_for_cpu() first.  The data may be stale
 * from the device's perspective.
 */
int process_dma_data(void)
{
  dma_addr_t handle;
  size_t buf_size = sizeof(cpu_buffer);

  /* Allocate DMA-coherent buffer */
  void *cpu_addr = dma_alloc_coherent(&test_dev, buf_size, &handle, 0);
  assert(cpu_addr != NULL);

  /*
   * BUG: Reading from the buffer without syncing first.
   * The device may have written new data that hasn't been
   * synchronized to the CPU cache.
   */
  uint8_t first_byte = ((uint8_t *)cpu_addr)[0];

  /*
   * The invariant: if the device wrote data to the DMA buffer,
   * the driver MUST have called dma_sync_single_for_cpu() before
   * reading it. This is a fundamental DMA coherence requirement.
   *
   * If dma_sync_called is 0 but dma_buffer[0] != 0, the driver
   * read stale data without syncing.
   */
  __ESBMC_assume(dma_buffer[0] == 0x42); /* Device wrote 0x42 */

  /*
   * Verify the invariant: either we synced, or the buffer was empty.
   * Since we assume dma_buffer[0] == 0x42, the only way this passes
   * is if dma_sync_called == 1.
   */
  assert(dma_sync_called == 1 || dma_buffer[0] == 0);

  dma_free_coherent(&test_dev, buf_size, cpu_addr, handle);
  return 0;
}

int main()
{
  test_dev.init_name = "test-cxl";
  memset(cpu_buffer, 0, sizeof(cpu_buffer));
  memset(dma_buffer, 0, sizeof(dma_buffer));
  dma_sync_called = 0;

  /*
   * Simulate: device writes 0x42 to the first byte of the DMA buffer.
   */
  dma_buffer[0] = 0x42;

  int ret = process_dma_data();
  (void)ret;
}
