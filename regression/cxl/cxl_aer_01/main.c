// PCIe AER (Advanced Error Reporting) test.
// Tests that the driver correctly handles PCIe error types:
// correctable, non-fatal, and fatal errors via AER.
// Expected: VERIFICATION SUCCESSFUL

#include <stdint.h>
#include <stddef.h>
#include <assert.h>

/* PCIe AER error severity */
enum aer_severity {
  AER_CORRECTABLE = 0,
  AER_NON_FATAL,
  AER_FATAL,
};

/* PCIe device with AER support */
struct pci_dev {
  uint16_t vendor;
  uint16_t device;
  enum aer_severity last_aer_error;
  uint64_t aer_correctable_count;
  uint64_t aer_non_fatal_count;
  uint64_t aer_fatal_count;
  int aer_enabled;
};

static struct pci_dev test_pci;

/* Enable AER on the device */
int pci_enable_aer(struct pci_dev *dev)
{
  assert(dev != NULL);
  dev->aer_enabled = 1;
  dev->last_aer_error = AER_CORRECTABLE;
  dev->aer_correctable_count = 0;
  dev->aer_non_fatal_count = 0;
  dev->aer_fatal_count = 0;
  return 0;
}

/* Inject a PCIe AER error */
void pci_aer_inject(struct pci_dev *dev, enum aer_severity severity)
{
  assert(dev != NULL);
  assert(dev->aer_enabled);

  dev->last_aer_error = severity;

  switch (severity)
  {
  case AER_CORRECTABLE:
    dev->aer_correctable_count++;
    break;
  case AER_NON_FATAL:
    dev->aer_non_fatal_count++;
    break;
  case AER_FATAL:
    dev->aer_fatal_count++;
    break;
  }
}

/* Read and clear AER error status */
enum aer_severity pci_aer_get_error(struct pci_dev *dev)
{
  assert(dev != NULL);
  enum aer_severity error = dev->last_aer_error;
  dev->last_aer_error = AER_CORRECTABLE; /* Clear */
  return error;
}

/* Handle AER error based on severity */
int pci_aer_handle_error(struct pci_dev *dev)
{
  assert(dev != NULL);

  enum aer_severity error = pci_aer_get_error(dev);

  switch (error)
  {
  case AER_CORRECTABLE:
    /* Log and continue */
    return 0;

  case AER_NON_FATAL:
    /* Attempt recovery */
    return 0;

  case AER_FATAL:
    /* Fatal: device must be reset */
    return -1;

  default:
    return 0;
  }
}

int main()
{
  test_pci.vendor = 0x1234;
  test_pci.device = 0x0001;
  test_pci.aer_enabled = 0;

  /* Enable AER */
  int ret = pci_enable_aer(&test_pci);
  assert(ret == 0);
  assert(test_pci.aer_enabled == 1);

  /* Test 1: Correctable AER error */
  pci_aer_inject(&test_pci, AER_CORRECTABLE);
  assert(pci_aer_handle_error(&test_pci) == 0);
  assert(test_pci.aer_correctable_count == 1);

  /* Test 2: Non-fatal AER error */
  pci_aer_inject(&test_pci, AER_NON_FATAL);
  assert(pci_aer_handle_error(&test_pci) == 0);
  assert(test_pci.aer_non_fatal_count == 1);

  /* Test 3: Fatal AER error */
  pci_aer_inject(&test_pci, AER_FATAL);
  assert(pci_aer_handle_error(&test_pci) == -1);
  assert(test_pci.aer_fatal_count == 1);

  /* Verify counts */
  assert(test_pci.aer_correctable_count == 1);
  assert(test_pci.aer_non_fatal_count == 1);
  assert(test_pci.aer_fatal_count == 1);
}
