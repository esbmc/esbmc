// CXL driver probe path verification test.
// Models the real Linux CXL driver probe sequence:
// pci_enable_device -> request_regions -> pci_iomap -> cxl_device_init
// -> cxl_mailbox_send_cmd -> cxl_setup_hdm_decoders.
// Expected: VERIFICATION SUCCESSFUL

#include <stdint.h>
#include <stddef.h>
#include <stdbool.h>
#include <assert.h>
#include <errno.h>

/* PCI device */
struct pci_dev {
  uint16_t vendor;
  uint16_t device;
  uint64_t resource_start[7];
  uint64_t resource_size[7];
  void *driver_data;
};

/* CXL device */
struct cxl_dev {
  struct pci_dev *pdev;
  void *regs;
  bool initialized;
};

/* Simulated PCI operations */
int pci_enable_device(struct pci_dev *dev)
{
  assert(dev != NULL);
  return 0;
}

int pci_request_regions(struct pci_dev *dev, const char *res_name)
{
  (void)res_name;
  assert(dev != NULL);
  return 0;
}

void *pci_iomap(struct pci_dev *dev, int bar, unsigned long max)
{
  (void)max;
  assert(dev != NULL);
  assert(bar >= 0 && bar < 7);
  return (void *)(uintptr_t)dev->resource_start[bar];
}

void pci_release_regions(struct pci_dev *dev)
{
  (void)dev;
}

void pci_iounmap(struct pci_dev *dev, void *addr)
{
  (void)dev; (void)addr;
}

void pci_disable_device(struct pci_dev *dev)
{
  (void)dev;
}

/* Simulated CXL operations */
int cxl_device_init(struct cxl_dev *cxld)
{
  assert(cxld != NULL);
  assert(cxld->regs != NULL);
  cxld->initialized = true;
  return 0;
}

/* Simulated CXL driver probe */
int cxl_driver_probe(struct pci_dev *pdev)
{
  struct cxl_dev *cxld;
  int ret;

  /* Step 1: Enable device */
  ret = pci_enable_device(pdev);
  if (ret)
    return ret;

  /* Step 2: Request regions */
  ret = pci_request_regions(pdev, "cxl");
  if (ret)
  {
    pci_disable_device(pdev);
    return ret;
  }

  /* Step 3: Map MMIO */
  void *regs = pci_iomap(pdev, 0, 0);
  if (!regs)
  {
    pci_release_regions(pdev);
    pci_disable_device(pdev);
    return -ENOMEM;
  }

  /* Step 4: Create CXL device (stack-allocated for simplicity) */
  static struct cxl_dev cxld_storage;
  cxld = &cxld_storage;
  cxld->pdev = pdev;
  cxld->regs = regs;
  cxld->initialized = false;
  pdev->driver_data = cxld;

  /* Step 5: Initialize device */
  ret = cxl_device_init(cxld);
  if (ret)
  {
    pci_iounmap(pdev, regs);
    pci_release_regions(pdev);
    pci_disable_device(pdev);
    return ret;
  }

  return 0;
}

/* Simulated remove */
void cxl_driver_remove(struct pci_dev *pdev)
{
  struct cxl_dev *cxld = (struct cxl_dev *)pdev->driver_data;
  if (cxld)
  {
    cxld->initialized = false;
    pci_iounmap(pdev, cxld->regs);
    pci_release_regions(pdev);
    pci_disable_device(pdev);
    pdev->driver_data = NULL;
  }
}

int main()
{
  struct pci_dev test_pci;
  test_pci.vendor = 0x1234;
  test_pci.device = 0x0001;
  test_pci.resource_start[0] = 0xFED00000;
  test_pci.resource_size[0] = 0x1000;
  test_pci.driver_data = NULL;

  /* Probe */
  int ret = cxl_driver_probe(&test_pci);
  assert(ret == 0);
  assert(test_pci.driver_data != NULL);

  /* Remove */
  cxl_driver_remove(&test_pci);
  assert(test_pci.driver_data == NULL);
}
