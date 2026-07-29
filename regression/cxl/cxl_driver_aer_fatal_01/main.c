// CXL PCIe AER fatal error handling test.
// Tests that the driver correctly handles a fatal AER error during
// device probe: enables AER, detects the error, and aborts initialization.
// Expected: VERIFICATION SUCCESSFUL
//
// Based on Linux kernel drivers/cxl/pci.c::cxl_pci_probe() which
// calls pci_enable_aer() and checks for fatal errors during probe.

#include <stdint.h>
#include <stddef.h>
#include <errno.h>
#include <assert.h>

#include <ubuntu20.04/kernel_5.15.0-76/include/linux/cxl.h>

/* Simulated PCI device */
struct pci_dev test_pci;
int aer_enabled = 0;
int fatal_error = 0;
int aer_probe_aborted = 0;

/* Override AER model for deterministic testing */
int pci_enable_aer(struct pci_dev *dev)
{
  (void)dev;
  aer_enabled = 1;
  return 0;
}

void pci_aer_clear(struct pci_dev *dev, int severity)
{
  (void)dev; (void)severity;
}

int pci_aer_get_first_error(struct pci_dev *dev, int *severity)
{
  (void)dev;
  assert(severity != NULL);

  if (fatal_error)
  {
    *severity = AER_FATAL;
    return 0;
  }
  *severity = AER_CORRECTABLE;
  return 0;
}

int pci_aer_clear_first_error(struct pci_dev *dev)
{
  (void)dev;
  if (fatal_error)
    return AER_FATAL;
  return AER_CORRECTABLE;
}

/*
 * Simulated CXL PCI probe — based on real kernel code pattern:
 *
 *   int cxl_pci_probe(struct pci_dev *pdev, ...)
 *   {
 *       int ret;
 *
 *       ret = pci_enable_aer(pdev);
 *       if (ret)
 *           return ret;
 *
 *       if (cxl_pci_check_fatal_error(pdev))
 *           return -EIO;
 *
 *       ... normal initialization ...
 *   }
 */
int cxl_pci_probe(struct pci_dev *pdev)
{
  int ret;
  int severity;

  /* Enable AER before probing */
  ret = pci_enable_aer(pdev);
  if (ret)
    return ret;

  /* Check for fatal AER errors */
  ret = pci_aer_get_first_error(pdev, &severity);
  if (ret == 0 && severity == AER_FATAL)
  {
    /* Fatal error: abort probe */
    aer_probe_aborted = 1;
    return -EIO;
  }

  /* Clear any non-fatal errors */
  if (ret == 0 && severity != AER_FATAL)
    pci_aer_clear(pdev, severity);

  return 0;
}

int main()
{
  test_pci.vendor = 0x1234;
  test_pci.device = 0x0001;
  aer_enabled = 0;
  fatal_error = 0;
  aer_probe_aborted = 0;

  /* Test 1: Normal probe with correctable error (should succeed) */
  fatal_error = 0;
  int ret = cxl_pci_probe(&test_pci);
  assert(ret == 0);
  assert(aer_enabled == 1);
  assert(aer_probe_aborted == 0);

  /* Test 2: Probe with fatal AER error (should abort) */
  fatal_error = 1;
  ret = cxl_pci_probe(&test_pci);
  assert(ret == -EIO);
  assert(aer_enabled == 1);
  assert(aer_probe_aborted == 1);

  __ESBMC_assert(aer_enabled == 1, "AER should be enabled before probe");
  __ESBMC_assert(aer_probe_aborted == 1,
                 "probe should abort on fatal AER error");
}
