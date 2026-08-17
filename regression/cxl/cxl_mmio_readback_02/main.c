// A register the driver has never written holds unknown power-on hardware
// state, so it must NOT be pinned to any particular value.  Asserting it
// reads as zero is the bug this test pins down.
// Expected: VERIFICATION FAILED

#include <stdint.h>
#include <stddef.h>
#include <assert.h>
#include <ubuntu20.04/kernel_5.15.0-76/include/linux/cxl.h>
#include <ubuntu20.04/kernel_5.15.0-76/include/asm/io.h>

int main()
{
  struct pci_dev dev;
  void *base = pci_iomap(&dev, 0, 4096);
  __ESBMC_assume(base != NULL);

  /* Writing one register must not define the contents of another. */
  writel(0x1u, base);

  __ESBMC_assert(
    readl((char *)base + 32) == 0,
    "unwritten MMIO register must not read as a fixed value");

  return 0;
}
