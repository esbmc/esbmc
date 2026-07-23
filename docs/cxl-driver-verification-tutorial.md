# CXL Driver Verification Tutorial for ESBMC

This tutorial shows how to verify CXL (Compute Express Link) device drivers
using ESBMC's operational model infrastructure.

## Prerequisites

- ESBMC built with the CXL driver verification support (commits
  `187ef9d7e1` through `17a1f1f8d1` on `feat/cxl-driver-verification`)
- Basic understanding of C programming and ESBMC's verification model

## Overview

ESBMC verifies CXL drivers by:

1. **Modeling hardware** as nondeterministic functions constrained by the CXL
   specification
2. **Separating memory spaces** (kernel, MMIO, DMA) to detect cross-space bugs
3. **Checking invariants** such as proper initialization ordering, DMA sync,
   and IRQ lifecycle

## Quick Start

### Running the included regression tests

```bash
# Run all CXL regression tests
ctest -L cxl

# Run a specific test
ctest -R "regression/cxl/cxl_mailbox_01"

# Run with a specific solver
ctest -L cxl --output-on-failure
```

### Running a custom test

```bash
esbmc my_cxl_driver.c
```

The CXL operational model is automatically linked when compiling C code that
includes the kernel headers under `src/c2goto/headers/`.

## Writing Your First CXL Test

### Step 1: Model your driver's data structures

```c
#include <stdint.h>
#include <stddef.h>
#include <assert.h>

/* Minimal CXL device structure */
struct cxl_dev {
  void *regs;           /* MMIO base */
  uint32_t command_count;
  int irq_registered;
};

static struct cxl_dev test_cxld;
```

### Step 2: Model the hardware operations

You can either use the built-in operational model functions or override them
for deterministic testing:

```c
/* Using built-in model (nondeterministic) */
#include <ubuntu20.04/kernel_5.15.0-76/include/asm/io.h>

void my_driver_function(void)
{
  /* readl/writel are modeled in cxl_driver.c */
  writel(0x1234, test_cxld.regs);
  uint32_t val = readl(test_cxld.regs);
}

/* Or override for deterministic testing */
uint32_t readl(const void *addr)
{
  return 0xDEADBEEF; /* Deterministic value */
}
```

### Step 3: Write verification assertions

```c
int main()
{
  test_cxld.regs = (void *)0xFED00000;

  /* Your driver code */
  my_driver_function();

  /* Verify invariants */
  assert(test_cxld.command_count >= 0);

  /* ESBMC-specific: assert that a bug condition never holds */
  __ESBMC_assert(test_cxld.irq_registered == 0 ||
                 test_cxld.command_count == 0,
                 "IRQ and command count invariant");

  return 0;
}
```

### Step 4: Create test.desc

```
CORE
main.c
^VERIFICATION SUCCESSFUL$
```

Or for a bug-detecting test:

```
CORE
main.c
^VERIFICATION FAILED$
```

## Common Bug Patterns

### 1. Missing DMA sync

```c
/* BUG: Reading DMA buffer without sync */
void process_dma_data(void)
{
  void *cpu_addr = dma_alloc_coherent(dev, size, &handle, GFP_KERNEL);
  /* BUG: Missing dma_sync_single_for_cpu()! */
  uint8_t data = ((uint8_t *)cpu_addr)[0];
  dma_free_coherent(dev, size, cpu_addr, handle);
}
```

**Fix:** Call `dma_sync_single_for_cpu()` before reading:

```c
void process_dma_data(void)
{
  void *cpu_addr = dma_alloc_coherent(dev, size, &handle, GFP_KERNEL);
  dma_sync_single_for_cpu(dev, handle, size, DMA_FROM_DEVICE);
  uint8_t data = ((uint8_t *)cpu_addr)[0];
  dma_free_coherent(dev, size, cpu_addr, handle);
}
```

### 2. Missing IRQ cleanup on remove

```c
/* BUG: free_irq not called in remove */
void my_driver_remove(struct pci_dev *pdev)
{
  /* Missing: free_irq(irq, dev_id); */
  pci_iounmap(pdev, regs);
  pci_release_regions(pdev);
}
```

### 3. Mailbox command without status check

```c
/* BUG: Using output without checking status */
int get_capabilities(void)
{
  struct cxl_mailbox_cmd cmd = { .opcode = CXL_MBOX_OP_GET_CAPABILITIES,
                                 .payload_out = buf,
                                 .payload_out_size = sizeof(buf) };
  cxl_mailbox_send_cmd(cxld, &cmd);
  /* BUG: No check of cmd.status! */
  return 0;
}
```

**Fix:** Check status before using output:

```c
int get_capabilities(void)
{
  struct cxl_mailbox_cmd cmd = { ... };
  int ret = cxl_mailbox_send_cmd(cxld, &cmd);
  if (ret) return ret;
  if (cmd.status) return -EIO;
  /* Safe to use cmd.payload_out */
  return 0;
}
```

### 4. HDM decoder overlap

```c
/* BUG: No overlap check between decoders */
int setup_decoders(void)
{
  cxl_setup_hdm_decoders(cxld, 0, 256MB, 0);
  cxl_setup_hdm_decoders(cxld, 128MB, 256MB, 1); /* Overlaps! */
}
```

**Fix:** Check for overlap before setting up decoders:

```c
int setup_decoders(void)
{
  cxl_setup_hdm_decoders(cxld, 0, 256MB, 0);
  /* Check: 128MB < 256MB (end of decoder 0) -> overlap! */
  if (new_base < existing_limit) return -EINVAL;
  cxl_setup_hdm_decoders(cxld, 256MB, 256MB, 1);
}
```

## Operational Model Reference

### MMIO Functions

| Function | Model |
|----------|-------|
| `readb/readw/readl/readq` | Return nondeterministic values from MMIO space |
| `writeb/writew/writel/writeq` | Store to MMIO space (read-back supported) |
| `wmb()/rmb()/mb()` | No-op (ordering via ESBMC thread interleaving) |
| `readl_relaxed/writel_relaxed` | Same as strict variants |

### DMA Functions

| Function | Model |
|----------|-------|
| `dma_alloc_coherent` | Allocates from DMA space, tracks usage |
| `dma_map_single` | Returns nondeterministic DMA address |
| `dma_sync_single_for_cpu` | Copies device data to CPU buffer |
| `dma_sync_single_for_device` | Copies CPU data to device buffer |

### IRQ Functions

| Function | Model |
|----------|-------|
| `request_irq` | Registers handler in IRQ table |
| `free_irq` | Removes handler from table |
| `esbmc_simulate_irq` | Invokes registered handler |

### CXL Functions

| Function | Model |
|----------|-------|
| `cxl_enumerate_ports` | Nondeterministic port enumeration |
| `cxl_device_init` | Sets INIT bit, simulates completion |
| `cxl_mailbox_send_cmd` | Nondeterministic success/failure |
| `cxl_setup_hdm_decoders` | Nondeterministic success/failure |
| `cxl_mem_attach/detach` | Lifecycle management |

## Debugging Tips

### 1. Use `--show-vcc` to see assertions

```bash
esbmc my_driver.c --show-vcc
```

### 2. Use `--goto-functions-only` to inspect the GOTO program

```bash
esbmc my_driver.c --goto-functions-only 2>&1 | grep -A20 "main"
```

### 3. Use counterexamples to understand failures

ESBMC outputs a detailed counterexample showing the state at each step:

```
[Counterexample]
State 1 file my_driver.c line 10 column 3 function main
  cxld->regs = 0x0  <-- NULL pointer!
```

### 4. Simplify with smaller test cases

When a test fails, create variants that isolate the problem:

```bash
# Test just the MMIO access
esbmc mmio_test.c

# Test just the DMA sync
esbmc dma_test.c

# Test the full driver
esbmc full_driver.c
```

## Further Reading

- [CXL Driver Verification Roadmap](./cxl-driver-verification-roadmap.md)
- ESBMC documentation: https://esbmc.org
- CXL 2.0 specification: https://cxlcomputingspec.org
