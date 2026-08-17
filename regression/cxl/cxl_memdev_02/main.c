// CXL memdev id allocator overflow: the driver indexes its device table
// with the raw allocator result.
// Expected: VERIFICATION FAILED (driver bug: unchecked ida_alloc_range())

#include <stdint.h>
#include <stddef.h>
#include <assert.h>
#include <ubuntu20.04/kernel_5.15.0-76/include/linux/cxl.h>
#include <ubuntu20.04/kernel_5.15.0-76/include/linux/cxlmem.h>

/* A driver-private table of the expanders it has brought up. */
#define DRIVER_MAX_MEMDEVS 4
static struct cxl_dev *driver_memdev_table[DRIVER_MAX_MEMDEVS];

int main()
{
  struct cxl_dev cxld;

  /*
   * BUG: cxl_memdev_id_alloc() models ida_alloc_range(), which yields
   * -ENOSPC once the id space is exhausted and otherwise any free minor
   * below CXL_MEM_MAX_DEVS. This driver checks neither the sign nor the
   * bound before using the result as a table index.
   */
  int id = cxl_memdev_id_alloc();
  driver_memdev_table[id] = &cxld;

  assert(driver_memdev_table[id] == &cxld);
  return 0;
}
