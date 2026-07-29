# CXL Driver Verification with ESBMC

This guide describes how to use ESBMC to verify CXL (Compute Express Link)
device drivers. It covers the operational models, regression test framework,
and how to extend the infrastructure for new CXL features.

## Overview

CXL is an industry-standard chip-to-chip interconnect that extends CPU memory
and I/O coherence to accelerators, storage, and other devices. Bugs in CXL
kernel-mode drivers can cause system crashes, data corruption, and security
vulnerabilities, making formal verification valuable.

ESBMC verifies CXL drivers by:

1. **Compiling driver code** via `c2goto` into a GOTO program
2. **Symbolically executing** the driver using the CXL operational models
3. **Encoding as SMT** and checking properties with an SMT solver (e.g., Z3)

## Operational Models

The CXL operational models are in `src/c2goto/library/cxl_driver.c` and are
automatically compiled into the c2goto library at build time. The models are
**non-deterministic** — they return values consistent with the CXL specification
while constraining the state space via `__ESBMC_assume()`.

### MMIO Access

Functions: `readb`, `readw`, `readl`, `readq`, `writeb`, `writew`, `writel`,
`writeq`, and relaxed variants.

- Reads return non-deterministic values
- Writes are stored in a 64 KB global MMIO space, so subsequent reads of the
  same address return the written value
- Out-of-bounds accesses return 0 (reads) or are no-ops (writes)

```c
writel(0x1, mmio_base + CXL_REGMAP_DEV_CTRL);
uint32_t echo = readl(mmio_base + CXL_REGMAP_DEV_CTRL);
// echo == 0x1 (written value, stored in MMIO space)
```

### PCI Device Model

Functions: `pci_get_device`, `pci_iomap`, `pci_enable_device`,
`pci_request_regions`, `pci_alloc_irq_vectors`, `pci_register_driver`, etc.

- PCI devices are tracked in an internal table (up to 16 devices)
- `pci_get_device()` may or may not find a device (nondeterministic)
- `pci_iomap()` maps BAR addresses into the MMIO space

### AER (Advanced Error Reporting)

Functions: `pci_enable_aer`, `pci_aer_clear`, `pci_aer_get_first_error`,
`pci_aer_clear_first_error`.

- Each PCI device has an AER capability structure tracking error severity
- `pci_aer_get_first_error()` returns non-deterministic severity from the
  available error types (correctable, non-fatal, fatal)

### IRQ (Interrupt Request)

Functions: `request_irq`, `free_irq`, `disable_irq`, `enable_irq`,
`esbmc_simulate_irq`.

- Up to 32 IRQ handlers registered per device
- `esbmc_simulate_irq()` invokes the registered handler with a given IRQ number

### DMA (Direct Memory Access)

Functions: `dma_alloc_coherent`, `dma_free_coherent`, `dma_map_single`,
`dma_unmap_single`, `dma_sync_single_for_cpu`, `dma_sync_single_for_device`,
`dma_set_mask`, `dma_set_coherent_mask`.

- DMA-coherent memory uses a separate 1 MB global array (`esbmc_dma_space`)
- Allows detection of stale-CPU-data bugs when the device writes without proper sync

### CXL Core API

Functions: `cxl_enumerate_ports`, `cxl_find_device`, `cxl_device_init`,
`cxl_device_exit`, `cxl_mailbox_send_cmd`, `cxl_get_security_state`,
`cxl_set_security`, `cxl_setup_hdm_decoders`, `cxl_driver_register`.

- `cxl_mailbox_send_cmd()` returns non-deterministic status codes
- `cxl_setup_hdm_decoders()` enforces 4KB alignment and the 8-decoder limit
  per CXL 2.0 specification

### Error Injection

Functions: `cxl_err_inject`, `cxl_err_get_count`.

- Injects correctable, non-fatal, or fatal errors into a CXL device
- Tracks error counts globally (per verification run)

## Writing Regression Tests

Each regression test lives in `regression/cxl/<suite>/` with two files:

### test.desc

```
CORE          # test class (CORE, KNOWNBUG, FUTURE, THOROUGH)
main.c        # source file
^VERIFICATION SUCCESSFUL$  # expected result (regex)
```

### main.c

The test file can use either:

**Option A: Override model functions** for deterministic testing. Define
your own implementations of model functions before `main()`. The compiler will
use your definitions instead of the operational model.

```c
/* Override for deterministic behavior */
int cxl_mailbox_send_cmd(struct cxl_dev *cxld, struct cxl_mailbox_cmd *cmd)
{
  cmd->status = 1;  /* simulate failure */
  return -EIO;
}
```

**Option B: Use the operational model** directly. The non-deterministic
behavior will be explored by ESBMC, making the test robust against all
valid hardware behaviors.

### Test Patterns

**Positive test** (verification should succeed):

```c
int main()
{
  /* Setup device state */
  /* Exercise driver code path */
  /* Assert invariants */
  __ESBMC_assert(result == expected, "Invariant violated");
}
```

**Bug-detecting test** (verification should fail):

```c
int main()
{
  /* Setup device state with a bug in the driver */
  /* The bug causes an invariant to be violated */
  __ESBMC_assert(!bug_condition, "Driver bug detected");
}
```

## Running Tests

```sh
# Run all CXL regression tests
ctest -j$(nproc) -L cxl --timeout 120

# Run a specific test
ctest -R "cxl_driver_hdm_align_01" --output-on-failure

# Run with verbose output and Z3 solver
ctest -R "cxl_driver_hdm_align_01" -V --output-on-failure --build-target esbmc
```

## Extending the Models

### Adding a New CXL Function

1. **Declare the function** in the appropriate header under
   `src/c2goto/headers/ubuntu20.04/kernel_5.15.0-76/include/linux/`
2. **Implement the model** in `src/c2goto/library/cxl_driver.c` in the
   relevant section (MMIO, PCI, CXL, DMA, IRQ, AER, etc.)
3. **Add a regression test** in `regression/cxl/`

The build system automatically picks up new `.c` files in the library
directory via `file(GLOB ... library/*.c)`.

### Example: Adding a New Function

```c
/* In the header (cxl.h or a new header): */
int cxl_my_new_function(struct cxl_dev *cxld, int param);

/* In cxl_driver.c: */
int cxl_my_new_function(struct cxl_dev *cxld, int param)
{
__ESBMC_HIDE:;
  assert(cxld != NULL);
  __ESBMC_assume(param >= 0 && param <= 255);

  int result = __VERIFIER_nondet_int();
  if (result != 0)
  {
    errno = EINVAL;
    return -1;
  }
  return 0;
}
```

## Known Limitations

| Limitation | Impact | Workaround |
|-----------|--------|------------|
| No ACPI _CCA/_CRS/_DSM modeling | Port enumeration is synthetic | Override models in tests |
| No hardware busy bits | Mailbox state machine is simplified | Use polling in tests |
| Command-completion interrupts not modeled | Mailbox completion relies on return values | Call `esbmc_simulate_irq()` manually |
| AER functions not in operational model (legacy) | See new functions | Use `pci_enable_aer()` etc. |
| Error injection not in operational model (legacy) | See new functions | Use `cxl_err_inject()` etc. |
| HDM alignment not enforced (legacy) | See new constraints | Use `cxl_setup_hdm_decoders()` with aligned addresses |
| No real Linux kernel driver source | Tests are synthetic | Write harnesses that exercise specific code paths |

## API Quick Reference

### MMIO
`readb`, `readw`, `readl`, `readq`, `writeb`, `writew`, `writel`, `writeq`,
`readl_relaxed`, `writel_relaxed`, `writesl`, `mb`, `wmb`, `rmb`,
`smp_mb`, `smp_wmb`, `smp_rmb`

### PCI
`pci_get_device`, `pci_get_bus_device`, `pci_put_device`,
`pci_enable_device`, `pci_disable_device`, `pci_request_regions`,
`pci_release_regions`, `pci_resource_start`, `pci_resource_end`,
`pci_resource_flags`, `pci_iomap`, `pci_iounmap`, `pci_enable_msi`,
`pci_disable_msi`, `pci_alloc_irq_vectors`, `pci_free_irq_vectors`,
`pci_register_driver`, `pci_unregister_driver`
`pci_read_config_byte`, `pci_read_config_word`, `pci_read_config_dword`,
`pci_write_config_byte`, `pci_write_config_word`, `pci_write_config_dword`

### AER
`pci_enable_aer`, `pci_aer_clear`, `pci_aer_get_first_error`,
`pci_aer_clear_first_error`

### IRQ
`request_irq`, `free_irq`, `disable_irq`, `enable_irq`,
`disable_irq_nosync`, `synchronize_irq`, `mask_irq`, `unmask_irq`,
`esbmc_simulate_irq`

### DMA
`dma_alloc_coherent`, `dma_free_coherent`, `dma_map_single`,
`dma_unmap_single`, `dma_sync_single_for_cpu`, `dma_sync_single_for_device`,
`dma_set_mask`, `dma_set_coherent_mask`

### CXL
`cxl_enumerate_ports`, `cxl_free_ports`, `cxl_find_device`,
`cxl_device_init`, `cxl_device_exit`, `cxl_mailbox_send_cmd`,
`cxl_read_dev_ctrl`, `cxl_write_dev_ctrl`, `cxl_read_dev_stat`,
`cxl_get_security_state`, `cxl_set_security`,
`cxl_setup_hdm_decoders`, `cxl_driver_register`, `cxl_driver_unregister`,
`cxl_err_inject`, `cxl_err_get_count`

### CXL Memory
`cxl_mem_attach`, `cxl_mem_detach`, `cxl_mem_flush`, `cxl_mem_enable`,
`cxl_mem_disable`, `cxl_mem_get_regions`, `cxl_mem_set_pmem_capacity`,
`cxl_mem_get_partition_state`, `cxl_mem_set_partition_state`
