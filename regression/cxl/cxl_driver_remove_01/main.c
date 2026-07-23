// CXL driver remove path verification test.
// Tests that the driver correctly cleans up all resources on remove:
// unregister IRQs -> disable device -> unmap MMIO -> release regions.
// Expected: VERIFICATION FAILED (driver bug: missing IRQ cleanup)

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>
#include <assert.h>

struct pci_dev {
  uint16_t vendor;
  uint16_t device;
  uint64_t resource_start[7];
  void *driver_data;
};

struct cxl_dev {
  struct pci_dev *pdev;
  void *regs;
  bool initialized;
  int irq_registered;
};

/* Simulated operations */
int pci_enable_device(struct pci_dev *dev) { (void)dev; return 0; }
int pci_request_regions(struct pci_dev *dev, const char *n) { (void)n; assert(dev); return 0; }
void *pci_iomap(struct pci_dev *dev, int bar, unsigned long m) { (void)m; assert(dev); assert(bar>=0 && bar<7); return (void*)(uintptr_t)dev->resource_start[bar]; }
void pci_release_regions(struct pci_dev *dev) { (void)dev; }
void pci_disable_device(struct pci_dev *dev) { (void)dev; }
void pci_iounmap(struct pci_dev *dev, void *addr) { (void)dev; (void)addr; }

int cxl_device_init(struct cxl_dev *cxld)
{
  assert(cxld); assert(cxld->regs);
  cxld->initialized = true;
  return 0;
}

int request_irq(unsigned int irq, void *handler, unsigned long flags, const char *name, void *dev_id)
{
  (void)irq; (void)handler; (void)flags; (void)name;
  struct cxl_dev *cxld = (struct cxl_dev *)dev_id;
  cxld->irq_registered = 1;
  return 0;
}

void free_irq(unsigned int irq, void *dev_id)
{
  (void)irq;
  struct cxl_dev *cxld = (struct cxl_dev *)dev_id;
  cxld->irq_registered = 0;
}

/*
 * BUG: This driver's remove function does NOT free the IRQ.
 * Real Linux drivers must call free_irq() before releasing other resources.
 */
void cxl_driver_remove(struct pci_dev *pdev)
{
  struct cxl_dev *cxld = (struct cxl_dev *)pdev->driver_data;
  if (!cxld)
    return;

  /* BUG: Missing free_irq() call! */

  if (cxld->regs)
    pci_iounmap(pdev, cxld->regs);
  pci_release_regions(pdev);
  pci_disable_device(pdev);
}

int main()
{
  struct pci_dev test_pci;
  struct cxl_dev test_cxld;

  test_pci.vendor = 0x1234;
  test_pci.device = 0x0001;
  test_pci.resource_start[0] = 0xFED00000;
  test_pci.driver_data = &test_cxld;

  test_cxld.pdev = &test_pci;
  test_cxld.regs = (void *)0xFED00000;
  test_cxld.initialized = true;
  test_cxld.irq_registered = 0;

  /* Simulate probe: register IRQ */
  request_irq(42, NULL, 0, "cxl-test", &test_cxld);
  assert(test_cxld.irq_registered == 1);

  /* Simulate remove */
  cxl_driver_remove(&test_pci);

  /*
   * The invariant: after remove, the IRQ should be freed.
   * Since the driver doesn't call free_irq, this assertion fails.
   */
  __ESBMC_assert(test_cxld.irq_registered == 0,
                 "IRQ not freed during driver remove");
}
