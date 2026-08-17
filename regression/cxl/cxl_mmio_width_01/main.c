// CXL register access at every width the model provides, plus the relaxed
// variants, block transfers, barriers and legacy port I/O.
//
// A register reads back what was written to it at the same width. The relaxed
// accessors differ from the ordered ones only in ordering, not in what they
// store, so an explicit barrier restores the guarantee.
// Expected: VERIFICATION SUCCESSFUL

#include <stddef.h>
#include <string.h>
#include <stdint.h>
#include <assert.h>
#include <ubuntu20.04/kernel_5.15.0-76/include/linux/pci.h>
#include <ubuntu20.04/kernel_5.15.0-76/include/asm/io.h>

#define CXL_BLOCK_WORDS 4

int main()
{
  struct pci_dev pdev;
  char *regs;

  memset(&pdev, 0, sizeof(pdev));
  regs = (char *)pci_iomap(&pdev, 0, 4096);
  if (!regs)
    return 0;

  writeb(0xA5, regs + 0x00);
  writew(0xBEEF, regs + 0x08);
  writel(0xDEADBEEFU, regs + 0x10);
  writeq(0x0123456789ABCDEFULL, regs + 0x18);
  wmb();

  assert(readb(regs + 0x00) == 0xA5);
  assert(readw(regs + 0x08) == 0xBEEF);
  assert(readl(regs + 0x10) == 0xDEADBEEFU);
  assert(readq(regs + 0x18) == 0x0123456789ABCDEFULL);
  rmb();

  /* Relaxed accessors store the same bytes; only the ordering is weaker. */
  writel_relaxed(0xC0FFEEU, regs + 0x20);
  mb();
  assert(readl_relaxed(regs + 0x20) == 0xC0FFEEU);

  /* A narrower write must not disturb the neighbouring register. */
  writeb(0x5A, regs + 0x00);
  assert(readl(regs + 0x10) == 0xDEADBEEFU);

  uint32_t out[CXL_BLOCK_WORDS] = {1, 2, 3, 4};
  uint32_t in[CXL_BLOCK_WORDS];
  writesl(regs + 0x40, out, CXL_BLOCK_WORDS);
  smp_wmb();
  readsl(regs + 0x40, in, CXL_BLOCK_WORDS);
  smp_rmb();
  for (int i = 0; i < CXL_BLOCK_WORDS; i++)
    assert(in[i] == out[i]);

  /* Legacy port I/O is unbacked hardware: it returns whatever the port gives,
     so the only thing to establish is that it is callable and ordered. */
  outb(0x01, 0xCF8);
  outw(0x0203, 0xCF8);
  outl(0x04050607U, 0xCFC);
  smp_mb();
  (void)inb(0xCFC);
  (void)inw(0xCFC);
  (void)inl(0xCFC);

  pci_iounmap(&pdev, regs);
  return 0;
}
