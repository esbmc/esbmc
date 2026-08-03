// CXL driver reading a 64-bit capability register with a 32-bit accessor.
// Expected: VERIFICATION FAILED (driver bug: register read at the wrong
// width, upper half silently dropped)

#include <stddef.h>
#include <string.h>
#include <stdint.h>
#include <assert.h>
#include <ubuntu20.04/kernel_5.15.0-76/include/linux/pci.h>
#include <ubuntu20.04/kernel_5.15.0-76/include/asm/io.h>

#define CXL_CAP_REG 0x18
#define CXL_CAP_VALUE 0x0000000100000000ULL /* only the upper half is set */

int main()
{
  struct pci_dev pdev;
  char *regs;

  memset(&pdev, 0, sizeof(pdev));
  regs = (char *)pci_iomap(&pdev, 0, 4096);
  if (!regs)
    return 0;

  writeq(CXL_CAP_VALUE, regs + CXL_CAP_REG);
  wmb();

  /*
   * BUG: a 64-bit register read with readl(). On a little-endian host the
   * low half reads back cleanly, which is what makes this survive testing --
   * every capability bit that lives above bit 31 is silently zero.
   */
  uint64_t cap = readl(regs + CXL_CAP_REG);
  assert(cap == CXL_CAP_VALUE);

  pci_iounmap(&pdev, regs);
  return 0;
}
