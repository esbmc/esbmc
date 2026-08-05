#include <linux/compiler-version.h>
#include <linux/kconfig.h>
#include <linux/compiler_types.h>

#include "drivers/cxl/core/pci.c"

/* pcie_link_speed_mbps() is left undefined, so ESBMC treats its result as an
 * unconstrained int. cxl_pci_get_latency() only guards bw < 0, then divides
 * by bw / BITS_PER_BYTE. */
int main(void)
{
  struct pci_dev *pdev = 0;
  long latency = cxl_pci_get_latency(pdev);
  (void)latency;
  return 0;
}
