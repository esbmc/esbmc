// CXL BAR inspection and topology lookup: register devices, query their
// resources, find one through the bridge, and release everything in order.
// Expected: VERIFICATION SUCCESSFUL

#include <stddef.h>
#include <string.h>
#include <assert.h>
#include <ubuntu20.04/kernel_5.15.0-76/include/linux/pci.h>
#include <ubuntu20.04/kernel_5.15.0-76/include/linux/irq.h>
#include <ubuntu20.04/kernel_5.15.0-76/include/linux/cxl.h>

#define CXL_VENDOR 0x8086
#define CXL_DEVICE 0x0d93

int main()
{
  struct pci_dev dev;
  struct cxl_host_bridge *bridge;

  /* Start from a known-empty table: enumeration state is global to the
     model, so a harness that cares about what it finds must say so. */
  esbmc_pci_reset_devices();

  memset(&dev, 0, sizeof(dev));
  dev.vendor = CXL_VENDOR;
  dev.device = CXL_DEVICE;
  assert(esbmc_pci_register_device(&dev) == 0);

  assert(pci_get_device(CXL_VENDOR, CXL_DEVICE, NULL) == &dev);
  /* Resuming from the only match ends the walk. */
  assert(pci_get_device(CXL_VENDOR, CXL_DEVICE, &dev) == NULL);
  /* A bus lookup can only return something now that the table is non-empty. */
  assert(pci_get_bus_device(0, 0, 0) == &dev);

  for (int bar = 0; bar < PCI_NUM_RESOURCES; bar++)
  {
    (void)pci_resource_start(&dev, bar);
    (void)pci_resource_end(&dev, bar);
    (void)pci_resource_flags(&dev, bar);
  }

  bridge = cxl_enumerate_ports();
  if (bridge)
  {
    struct cxl_dev *cxld = cxl_find_device(bridge, CXL_VENDOR, CXL_DEVICE);
    /* A bridge with no downstream devices has nothing to find. */
    if (bridge->num_devices == 0)
      assert(cxld == NULL);
    else
      assert(cxld != NULL && cxld->regs != NULL);

    cxl_free_ports(bridge);
  }

  disable_irq_nosync(dev.irq);
  esbmc_pci_reset_devices();
  return 0;
}
