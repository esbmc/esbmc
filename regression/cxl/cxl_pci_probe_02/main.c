// CXL PCI probe whose BAR-type rejection returns directly instead of entering
// the unwind ladder.
// Expected: VERIFICATION FAILED (driver bug: leak on the error path, CWE-401)

#include <stddef.h>
#include <assert.h>
#include <errno.h>
#include <ubuntu20.04/kernel_5.15.0-76/include/linux/pci.h>
#include <ubuntu20.04/kernel_5.15.0-76/include/linux/slab.h>
#include <ubuntu20.04/kernel_5.15.0-76/include/linux/gfp.h>

#define CXL_VENDOR_INTEL 0x8086
#define CXL_DEVICE_MEM 0x0d93
#define CXL_REGS_BAR 0

struct cxl_pci_priv
{
  void *regs;
  struct pci_dev *pdev;
};

static int cxl_pci_probe(struct pci_dev *pdev)
{
  struct cxl_pci_priv *priv;
  int rc;

  priv = kmalloc(sizeof(*priv), GFP_KERNEL);
  if (!priv)
    return -ENOMEM;
  priv->pdev = pdev;

  rc = pci_enable_device(pdev);
  if (rc)
    goto err_free;

  rc = pci_request_regions(pdev, "cxl_pci");
  if (rc)
    goto err_disable;

  /*
   * BUG: this rejection is the one path out of probe() that does not join the
   * unwind ladder. It reads like a validation guard rather than a failure, so
   * the early return looks harmless -- but priv is already allocated, and the
   * regions and the enable are already taken.
   */
  if (!(pci_resource_flags(pdev, CXL_REGS_BAR) & PCI_BASE_ADDRESS_SPACE_MEMORY))
    return -ENXIO;

  priv->regs = pci_iomap(pdev, CXL_REGS_BAR, 4096);
  if (!priv->regs)
  {
    rc = -ENOMEM;
    goto err_release;
  }

  pci_iounmap(pdev, priv->regs);
  pci_release_regions(pdev);
  pci_disable_device(pdev);
  kfree(priv);
  return 0;

err_release:
  pci_release_regions(pdev);
err_disable:
  pci_disable_device(pdev);
err_free:
  kfree(priv);
  return rc;
}

int main()
{
  struct pci_dev dev;
  struct pci_driver drv;

  dev.vendor = CXL_VENDOR_INTEL;
  dev.device = CXL_DEVICE_MEM;
  assert(esbmc_pci_register_device(&dev) == 0);

  drv.name = "cxl_pci";
  drv.id_table = NULL;
  drv.probe = NULL;
  drv.remove = NULL;
  if (pci_register_driver(&drv))
    return 0;

  struct pci_dev *found = pci_get_device(CXL_VENDOR_INTEL, CXL_DEVICE_MEM, NULL);
  assert(found == &dev);

  cxl_pci_probe(found);

  pci_put_device(found);
  pci_unregister_driver(&drv);
  return 0;
}
