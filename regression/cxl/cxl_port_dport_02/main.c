// CXL dangling dport reference: the driver keeps using a dport pointer it
// cached before the parent port dropped that dport.
// Expected: VERIFICATION FAILED (driver bug: use-after-free)

#include <stdint.h>
#include <stddef.h>
#include <errno.h>
#include <assert.h>
#include <ubuntu20.04/kernel_5.15.0-76/include/linux/cxl.h>

int main()
{
  struct cxl_port port;
  struct pci_dev d0;

  if (cxl_dport_add(&port, &d0, 0) != 0)
    return 0;

  struct cxl_dport *dp = cxl_dport_find(&port, 0);
  if (dp == NULL)
    return 0;

  /*
   * BUG: the dport belongs to the port, and the port frees it here. In
   * Linux the same thing happens under devm cleanup. The driver caches
   * the pointer across that boundary and dereferences it afterwards.
   */
  cxl_dport_remove(&port, 0);
  assert(dp->id == 0);

  return 0;
}
