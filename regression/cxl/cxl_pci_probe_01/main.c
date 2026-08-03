// CXL PCI probe with a fully unwound error path: every acquisition taken
// before a failure is released again, in reverse order.
// Expected: VERIFICATION SUCCESSFUL

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

  /* A CXL register block must be memory-mapped; an I/O-space BAR means the
     device is not what the id table claimed. */
  if (!(pci_resource_flags(pdev, CXL_REGS_BAR) & PCI_BASE_ADDRESS_SPACE_MEMORY))
  {
    rc = -ENXIO;
    goto err_release;
  }

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
  /* Registered above, so enumeration must find it, and must find that one. */
  assert(found == &dev);

  cxl_pci_probe(found);

  pci_put_device(found);
  pci_unregister_driver(&drv);
  return 0;
}
