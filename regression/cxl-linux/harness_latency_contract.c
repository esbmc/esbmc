#include <linux/compiler-version.h>
#include <linux/kconfig.h>
#include <linux/compiler_types.h>

#include "drivers/cxl/core/pci.c"

int __VERIFIER_nondet_int(void);

/* Model the callee's real contract: pcie_dev_speed_mbps() (drivers/pci/pci.h)
 * returns -EINVAL or one of the tabulated PCIe speeds, and
 * pcie_link_speed_mbps() otherwise forwards a negative error. */
int pcie_link_speed_mbps(struct pci_dev *pdev)
{
  int r = __VERIFIER_nondet_int();
  __ESBMC_assume(
    r < 0 || r == 2500 || r == 5000 || r == 8000 || r == 16000 || r == 32000 ||
    r == 64000);
  return r;
}

int main(void)
{
  struct pci_dev *pdev = 0;
  long latency = cxl_pci_get_latency(pdev);
  (void)latency;
  return 0;
}
