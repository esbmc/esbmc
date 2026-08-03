// CXL host-bridge enumeration: walk the downstream devices, then release the
// whole topology.
//
// cxl_enumerate_ports() reports a bridge that may have no downstream devices
// at all, in which case it carries no device array. A walk that assumes
// otherwise is walking a topology the enumeration never promised.
// Expected: VERIFICATION SUCCESSFUL

#include <stddef.h>
#include <assert.h>
#include <ubuntu20.04/kernel_5.15.0-76/include/linux/cxl.h>


int main()
{
  struct cxl_host_bridge *bridge;
  unsigned int seen = 0;

  bridge = cxl_enumerate_ports();
  if (!bridge)
    return 0;

  assert(bridge->num_devices <= CXL_MAX_DOWNSTREAM_PORTS);

  /* An empty bridge has no device array at all. */
  if (bridge->num_devices == 0)
  {
    cxl_free_ports(bridge);
    return 0;
  }
  assert(bridge->devices != NULL);

  for (unsigned int i = 0; i < bridge->num_devices; i++)
  {
    struct cxl_dev *cxld = &bridge->devices[i];

    assert(cxld->dev_type >= CXL_TYPE_FPMEM && cxld->dev_type <= CXL_TYPE_RAM);
    assert(cxld->regs != NULL);
    seen++;
  }

  assert(seen <= bridge->num_devices);

  cxl_free_ports(bridge);
  return 0;
}
