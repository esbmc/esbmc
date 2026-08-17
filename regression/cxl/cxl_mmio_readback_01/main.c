// MMIO read-back: a register returns the value the driver last wrote to it.
// Expected: VERIFICATION SUCCESSFUL

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

  writel(0xDEADBEEFu, base);
  assert(readl(base) == 0xDEADBEEFu);

  /* A later write to the same register wins. */
  writel(0x1u, base);
  assert(readl(base) == 0x1u);

  /* Read-back is per-width and per-address. */
  writeb(0xABu, (char *)base + 64);
  assert(readb((char *)base + 64) == 0xABu);

  writeq(0x1122334455667788ull, (char *)base + 128);
  assert(readq((char *)base + 128) == 0x1122334455667788ull);

  /* Distinct registers do not alias. */
  writel(0xAAAAAAAAu, (char *)base + 256);
  writel(0xBBBBBBBBu, (char *)base + 260);
  assert(readl((char *)base + 256) == 0xAAAAAAAAu);
  assert(readl((char *)base + 260) == 0xBBBBBBBBu);

  return 0;
}
