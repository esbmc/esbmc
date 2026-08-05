#include <linux/compiler-version.h>
#include <linux/kconfig.h>
#include <linux/compiler_types.h>

#include "drivers/cxl/core/pci.c"

int __VERIFIER_nondet_int(void);
unsigned short __VERIFIER_nondet_ushort(void);
unsigned int __VERIFIER_nondet_uint(void);

/* PCI config space is hardware state: every read yields an unconstrained
 * value, and may fail. */
int pci_read_config_word(const struct pci_dev *dev, int where, u16 *val)
{
  *val = __VERIFIER_nondet_ushort();
  return __VERIFIER_nondet_int();
}

int pci_read_config_dword(const struct pci_dev *dev, int where, u32 *val)
{
  *val = __VERIFIER_nondet_uint();
  return __VERIFIER_nondet_int();
}

int main(void)
{
  static struct pci_dev pdev;
  static struct cxl_dev_state cxlds;
  struct cxl_endpoint_dvsec_info info = {};

  cxlds.dev = &pdev.dev;
  cxlds.cxl_dvsec = __VERIFIER_nondet_int();

  int rc = cxl_dvsec_rr_decode(&cxlds, &info);
  (void)rc;

  /* dvsec_range[] holds 2 entries and the loop writes one per iteration, so
   * the hdm_count > 2 rejection is what keeps the store in bounds. */
  assert(info.ranges <= 1); /* liveness: 2 ranges are reachable */
  return 0;
}
