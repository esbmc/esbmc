#include <linux/compiler-version.h>
#include <linux/kconfig.h>
#include <linux/compiler_types.h>

#include "drivers/cxl/core/pci.c"

unsigned long __VERIFIER_nondet_ulong(void);
unsigned char __VERIFIER_nondet_uchar(void);

#define BUF_SZ 8

int main(void)
{
  unsigned char buf[BUF_SZ];
  for (int i = 0; i < BUF_SZ; i++)
    buf[i] = __VERIFIER_nondet_uchar();

  size_t size = __VERIFIER_nondet_ulong();
  __ESBMC_assume(size <= BUF_SZ);

  unsigned char sum = cdat_checksum(buf, size);

  /* The checksum is the truncated sum of the bytes read; with size == 0 no
   * byte is read and the result must be 0. */
  if (size == 0)
    assert(sum == 0);
  return 0;
}
