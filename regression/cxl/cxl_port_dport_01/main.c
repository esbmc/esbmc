// CXL downstream port traversal: dports register under their parent port,
// are found by id, and disappear on removal.
// Expected: VERIFICATION SUCCESSFUL

#include <stdint.h>
#include <stddef.h>
#include <errno.h>
#include <assert.h>
#include <ubuntu20.04/kernel_5.15.0-76/include/linux/cxl.h>

int main()
{
  struct cxl_port port;
  struct pci_dev d0;
  struct pci_dev d1;

  assert(cxl_dport_count(&port) == 0);

  /* Registration allocates, so it may legitimately fail. */
  if (cxl_dport_add(&port, &d0, 0) != 0)
    return 0;
  if (cxl_dport_add(&port, &d1, 1) != 0)
    return 0;
  assert(cxl_dport_count(&port) == 2);

  /* A second dport claiming a live id is refused. */
  assert(cxl_dport_add(&port, &d1, 1) == -EBUSY);

  struct cxl_dport *dp = cxl_dport_find(&port, 1);
  assert(dp != NULL);
  assert(dp->id == 1);
  assert(dp->dport_dev == &d1);
  assert(dp->port == &port);

  /* An id that was never registered is absent. */
  assert(cxl_dport_find(&port, 7) == NULL);

  cxl_dport_remove(&port, 1);
  assert(cxl_dport_find(&port, 1) == NULL);
  assert(cxl_dport_count(&port) == 1);

  /* The surviving dport is untouched by its sibling's removal. */
  struct cxl_dport *survivor = cxl_dport_find(&port, 0);
  assert(survivor != NULL);
  assert(survivor->id == 0);

  cxl_dport_remove(&port, 0);
  assert(cxl_dport_count(&port) == 0);

  return 0;
}
