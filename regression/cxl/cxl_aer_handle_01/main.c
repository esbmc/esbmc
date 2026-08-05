// CXL PCIe AER error handling: enable reporting, then drain the first-error
// register until it is clear.
//
// The AER state is per-device, so a driver managing two endpoints must not
// see one device's errors through the other's handle.
// Expected: VERIFICATION SUCCESSFUL

#include <stddef.h>
#include <assert.h>
#include <errno.h>
#include <string.h>
#include <ubuntu20.04/kernel_5.15.0-76/include/linux/pci.h>
#include <ubuntu20.04/kernel_5.15.0-76/include/linux/cxl.h>

int main()
{
  struct pci_dev dev_a, dev_b;
  int severity;

  /* Fabricated devices: a real one arrives from the bus zeroed, and the AER
     fields decide whether reporting is on. */
  memset(&dev_a, 0, sizeof(dev_a));
  memset(&dev_b, 0, sizeof(dev_b));

  dev_a.vendor = 0x8086;
  dev_a.device = 0x0d93;
  dev_b.vendor = 0x8086;
  dev_b.device = 0x0d94;

  /* Reporting is off until enabled: querying an unenabled device must say so
     rather than inventing a severity. */
  assert(pci_aer_get_first_error(&dev_a, &severity) == -ENODEV);

  if (pci_enable_aer(&dev_a))
    return 0;

  /* dev_b is still unenabled, and enabling dev_a must not have enabled it. */
  assert(pci_aer_get_first_error(&dev_b, &severity) == -ENODEV);

  if (pci_enable_aer(&dev_b))
    return 0;

  assert(pci_aer_get_first_error(&dev_a, &severity) == 0);
  assert(severity >= AER_CORRECTABLE && severity <= AER_FATAL);

  pci_aer_clear(&dev_a, severity);

  int cleared = pci_aer_clear_first_error(&dev_a);
  assert(cleared == severity);

  /* Clearing dev_a leaves dev_b's own state reachable and independent. */
  assert(pci_aer_get_first_error(&dev_b, &severity) == 0);

  pci_aer_clear(&dev_b, AER_FATAL);
  assert(pci_aer_clear_first_error(&dev_b) == AER_FATAL);

  return 0;
}
