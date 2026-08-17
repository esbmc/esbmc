// CXL PCI device enumeration and BAR mapping test.
// Tests that PCI devices are enumerated and BARs are mapped correctly.
// Expected: VERIFICATION SUCCESSFUL

#include <stdint.h>
#include <stddef.h>
#include <assert.h>

/* PCI types */
typedef unsigned short u16;
typedef unsigned int u32;
typedef long resource_size_t;

#define PCI_NUM_RESOURCES_WITH_ROM 7

struct pci_dev {
  u16 vendor;
  u16 device;
  resource_size_t resource_start[PCI_NUM_RESOURCES_WITH_ROM];
  resource_size_t resource_size[PCI_NUM_RESOURCES_WITH_ROM];
  u32 resource_flags[PCI_NUM_RESOURCES_WITH_ROM];
  u32 irq;
  void *driver_data;
};

/* Simulated PCI device table */
static struct pci_dev pci_devices[4];
static int pci_count = 0;

struct pci_dev *pci_get_device(u16 vendor, u16 device, struct pci_dev *from)
{
  (void)from;
  if (pci_count == 0) return NULL;
  /* Return first matching device */
  for (int i = 0; i < pci_count; i++)
  {
    if ((vendor == 0 || pci_devices[i].vendor == vendor) &&
        (device == 0 || pci_devices[i].device == device))
    {
      return &pci_devices[i];
    }
  }
  return NULL;
}

resource_size_t pci_resource_start(struct pci_dev *dev, int bar)
{
  assert(dev != NULL);
  assert(bar >= 0 && bar < PCI_NUM_RESOURCES_WITH_ROM);
  return dev->resource_start[bar];
}

resource_size_t pci_resource_end(struct pci_dev *dev, int bar)
{
  assert(dev != NULL);
  assert(bar >= 0 && bar < PCI_NUM_RESOURCES_WITH_ROM);
  return dev->resource_start[bar] + dev->resource_size[bar] - 1;
}

void *pci_iomap(struct pci_dev *dev, int bar, unsigned long max)
{
  assert(dev != NULL);
  assert(bar >= 0 && bar < PCI_NUM_RESOURCES_WITH_ROM);
  resource_size_t start = pci_resource_start(dev, bar);
  if (start == 0) return NULL;
  /* Return a non-NULL pointer to simulate mapping */
  return (void *)(uintptr_t)start;
}

int main()
{
  /* Simulate a CXL device at vendor=0x1234, device=0x0001 */
  pci_devices[0].vendor = 0x1234;
  pci_devices[0].device = 0x0001;
  pci_devices[0].resource_start[0] = 0xFED00000;
  pci_devices[0].resource_size[0] = 0x1000;
  pci_count = 1;

  /* Enumerate */
  struct pci_dev *dev = pci_get_device(0x1234, 0x0001, NULL);
  assert(dev != NULL);
  assert(dev->vendor == 0x1234);
  assert(dev->device == 0x0001);

  /* Check BAR0 */
  resource_size_t start = pci_resource_start(dev, 0);
  resource_size_t end = pci_resource_end(dev, 0);
  assert(start == 0xFED00000);
  assert(end == 0xFED00FFF);

  /* Map BAR0 */
  void *mmio = pci_iomap(dev, 0, 0);
  assert(mmio != NULL);
  assert((uintptr_t)mmio == 0xFED00000);
}
