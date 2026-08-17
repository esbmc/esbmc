// CXL PCI device not found — accessing BAR on absent device.
// Tests that the driver handles missing devices gracefully.
// Expected: VERIFICATION FAILED (driver bug: uses NULL device pointer)

#include <stdint.h>
#include <stddef.h>
#include <assert.h>

typedef unsigned short u16;
typedef long resource_size_t;

#define PCI_NUM_RESOURCES_WITH_ROM 7

struct pci_dev {
  u16 vendor;
  u16 device;
  resource_size_t resource_start[PCI_NUM_RESOURCES_WITH_ROM];
  resource_size_t resource_size[PCI_NUM_RESOURCES_WITH_ROM];
};

static struct pci_dev pci_devices[4];
static int pci_count = 0;

struct pci_dev *pci_get_device(u16 vendor, u16 device, struct pci_dev *from)
{
  (void)vendor; (void)device; (void)from;
  (void)pci_count; (void)pci_devices;
  return NULL; /* No device found */
}

resource_size_t pci_resource_start(struct pci_dev *dev, int bar)
{
  assert(dev != NULL);
  assert(bar >= 0 && bar < PCI_NUM_RESOURCES_WITH_ROM);
  return dev->resource_start[bar];
}

/*
 * BUG: This driver does not check if pci_get_device() returned NULL
 * before accessing the device's BARs.
 */
int setup_cxl_device(void)
{
  struct pci_dev *dev = pci_get_device(0x1234, 0x0001, NULL);

  /* BUG: No NULL check! */
  resource_size_t start = pci_resource_start(dev, 0);
  (void)start;

  return 0;
}

int main()
{
  int ret = setup_cxl_device();
  (void)ret;
}
