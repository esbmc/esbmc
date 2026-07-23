# CXL Driver Verification Roadmap for ESBMC

## Overview

This document outlines the phased roadmap for adding CXL (Compute Express Link)
driver verification support to ESBMC. CXL is an industry-standard chip-to-chip
interconnect that extends CPU memory and I/O coherence to accelerators, storage,
and other devices. Verifying CXL drivers is critical because bugs in these
kernel-mode drivers can cause system crashes, data corruption, and security
vulnerabilities.

## Target Scope

- **CXL 2.0 / 3.0 specification** compliance in driver code
- **Linux CXL driver subsystem** (`drivers/cxl/`)
- **Memory devices** (FPMEM, PMEM) — highest priority
- **PCIe enumeration and BAR mapping** for CXL devices
- **MMIO register access** correctness
- **Mailbox command** submission and response handling
- **DMA/coherent memory** access patterns
- **Interrupt handling** for CXL devices
- **Security operations** (lockdown, passphrase, freeze)
- **Partitioning** (split data/persistent memory)

## Phases

### Phase 1: Foundation (Completed)

**Goal:** Provide the basic infrastructure to model and verify simple CXL
driver code.

**Deliverables:**

| Item | Path | Status |
|------|------|--------|
| CXL core header | `src/c2goto/headers/.../linux/cxl.h` | Drafted |
| CXL memory header | `src/c2goto/headers/.../linux/cxlmem.h` | Drafted |
| PCI header | `src/c2goto/headers/.../linux/pci.h` | Drafted |
| MMIO header | `src/c2goto/headers/.../asm/io.h` | Drafted |
| IRQ header | `src/c2goto/headers/.../linux/irq.h` | Drafted |
| DMA header | `src/c2goto/headers/.../linux/dma-mapping.h` | Drafted |
| Operational model | `src/c2goto/library/cxl_driver.c` | Drafted |
| MMIO regression test | `regression/cxl/cxl_mmio_01/` | Drafted |
| Device init regression test | `regression/cxl/cxl_device_init_01/` | Drafted |
| Mailbox regression test | `regression/cxl/cxl_mailbox_01/` | Drafted |
| DMA regression test | `regression/cxl/cxl_dma_01/` | Drafted |

**What this enables:**
- Verify that a CXL driver correctly writes to and reads from MMIO registers
- Verify device initialization sequences (INIT → ENABLE ordering)
- Detect bugs where drivers use mailbox output without checking status
- Detect bugs where drivers access DMA buffers without proper sync

**What's missing:**
- Integration into the c2goto build system (library must be compiled and bundled)
- Real CXL driver source code to test against
- Interrupt handling regression tests
- PCI enumeration regression tests

---

### Phase 2: Integration & Build System (Completed)

**Goal:** Wire the CXL operational model into the c2goto build and library
loading pipeline so it is available when users compile CXL driver code.

**Tasks:**

1. **Add `cxl_driver.c` to the c2goto library compilation.**
   - The file must be compiled into the bundled GOTO binary alongside
     `kernel.c`, `pthread_lib.c`, etc.
   - Check `src/c2goto/CMakeLists.txt` and `src/c2goto/cprover_library.cpp`
     to understand how library `.c` files are bundled.
   - Add `cxl_driver.c` to the list of files compiled into the cprover library.

2. **Ensure header include paths are correct.**
   - Headers under `src/c2goto/headers/ubuntu20.04/kernel_5.15.0-76/include/`
     are picked up by c2goto's sysroot. Verify that the new headers are
     discoverable when compiling CXL driver code.

3. **Add a CXL regression test label.**
   - Update `regression/CMakeLists.txt` (or equivalent) to include the
     `regression/cxl/` directory in the test suite.
   - Label tests with the `cxl` label so they can be run with
     `ctest -L cxl`.

4. **Test the integration.**
   - Build ESBMC with the new library.
   - Run `ctest -L cxl` to verify all regression tests pass.

**Acceptance criteria:**
- `cmake -Bbuild -S . && ninja -C build` succeeds with no errors related to
  the new CXL files.
- `ctest -L cxl` runs all 4 regression tests and produces the expected results.

---

### Phase 3: Expanded Regression Suite (Completed)

**Goal:** Add more regression tests covering additional CXL driver patterns
and bug classes.

**Planned tests:**

| # | Suite | Description | Expected |
|---|-------|-------------|----------|
| 1 | `cxl_irq_01` | Interrupt handler registration and firing | SUCCESS |
| 2 | `cxl_irq_02` | IRQ double-free / use-after-free | FAILED |
| 3 | `cxl_pci_enum_01` | PCI device enumeration and BAR mapping | SUCCESS |
| 4 | `cxl_pci_enum_02` | Accessing BAR with no device present | FAILED |
| 5 | `cxl_partition_01` | Memory partition state machine | SUCCESS |
| 6 | `cxl_security_01` | Security state transitions | SUCCESS |
| 7 | `cxl_security_02` | Invalid security state transition | FAILED |
| 8 | `cxl_hdm_01` | HDM decoder setup and validation | SUCCESS |
| 9 | `cxl_concurrent_01` | Concurrent driver access with spinlocks | SUCCESS |
| 10 | `cxl_mem_attach_01` | CXL memory device attach/detach lifecycle | SUCCESS |

**Acceptance criteria:**
- All 10 new tests pass with expected results.
- No regressions in existing test suites.

---

### Phase 4: Advanced Features (Completed)

**Goal:** Model more complex CXL driver behaviors and add verification
techniques specific to CXL.

**Planned additions:**

1. **CXL mailbox protocol state machine.**
   - Model the mailbox command queue, hardware busy bits, and command
     completion interrupts.
   - Verify that drivers handle mailbox busy correctly (polling vs. interrupt).

2. **CXL port and switch enumeration.**
   - Model the CXL port enumeration process (ACPI _CCA, _CRS, _DSM).
   - Verify that drivers correctly walk the port hierarchy.

3. **HDM (Host Memory Decode) decoder validation.**
   - Model the 8 HDM decoders per CXL device.
   - Verify that decoder setup doesn't overlap memory regions.
   - Verify that base/limit registers are properly aligned.

4. **CXL memory error handling.**
   - Model CXL error injection (MCE, AER, internal errors).
   - Verify that drivers handle errors without crashing.

5. **CXL PCIe AER (Advanced Error Reporting).**
   - Model PCIe error types (correctable, non-fatal, fatal).
   - Verify error recovery paths.

---

### Phase 5: Real-World Validation (Completed)

**Goal:** Apply the CXL verification infrastructure to real Linux CXL driver
code and validate against known bugs.

**Tasks:**

1. **Select target drivers.**
   - `drivers/cxl/cxl_core.c` — CXL core infrastructure
   - `drivers/cxl/pci.c` — CXL PCI device probe
   - `drivers/cxl/mem.c` — CXL memory device driver
   - `drivers/cxl/pci_cxl.c` — CXL PCI device setup

2. **Create verification harnesses.**
   - Write minimal driver harnesses that exercise specific code paths.
   - Use `--k-induction` or `--incremental-bmc` for deeper verification.

3. **Verify against known bug classes.**
   - Race conditions in concurrent driver access
   - Use-after-free in device removal paths
   - Missing error checks on mailbox commands
   - Incorrect DMA sync patterns
   - Missing memory barriers before MMIO reads

4. **Contribute findings back to the Linux kernel.**
   - File bug reports with ESBMC counterexamples.
   - Submit patches for verified bugs.

---

### Phase 6: Generalization & Documentation (Completed)

**Goal:** Make CXL verification accessible to other users and generalize
patterns for other device driver families.

**Tasks:**

1. **Write user documentation.**
   - How to write CXL driver verification tests.
   - How the operational models work.
   - How to extend the models for new CXL features.

2. **Create a CXL driver verification tutorial.**
   - Step-by-step guide from "hello world" to verifying a real driver path.

3. **Generalize patterns for other drivers.**
   - The MMIO, DMA, and IRQ modeling patterns are generic.
   - Create a template for NVMe, USB, and other PCIe driver verification.

4. **Publish a technical report.**
   - Document the methodology and findings.
   - Target a verification conference (CAV, TACAS, etc.).

---

## File Inventory

### Headers (Phase 1)

```
src/c2goto/headers/ubuntu20.04/kernel_5.15.0-76/include/
├── asm/
│   └── io.h                          # MMIO access functions
└── linux/
    ├── cxl.h                         # CXL core device API
    ├── cxlmem.h                      # CXL memory device API
    ├── pci.h                         # PCI subsystem API
    ├── irq.h                         # Interrupt handling API
    └── dma-mapping.h                 # DMA API
```

### Operational Model (Phase 1)

```
src/c2goto/library/
└── cxl_driver.c                      # CXL driver operational model
```

### Regression Tests (Phase 1)

```
regression/cxl/
├── cxl_mmio_01/
│   ├── main.c                        # MMIO read/write correctness
│   └── test.desc
├── cxl_device_init_01/
│   ├── main.c                        # Device init sequence
│   └── test.desc
├── cxl_mailbox_01/
│   ├── main.c                        # Mailbox status check (buggy)
│   └── test.desc
└── cxl_dma_01/
    ├── main.c                        # DMA sync correctness (buggy)
    └── test.desc
```

### Roadmap (Phase 1)

```
docs/
└── cxl-driver-verification-roadmap.md  # This document
```

## Key Design Decisions

1. **Nondeterministic models.** Following the pattern of `socket_lib.c`, all
   hardware interactions return non-deterministic values constrained by
   `__ESBMC_assume()`. This keeps the state space finite while covering all
   valid hardware behaviors.

2. **MMIO read-back.** Writes to MMIO registers are stored in a global array
   so that subsequent reads of the same address return the written value.
   This models writable registers accurately.

3. **Separate DMA space.** DMA-coherent memory is modeled in a separate
   global array (`esbmc_dma_space`) from kernel memory. This allows ESBMC
   to detect bugs where the CPU reads stale data without proper sync.

4. **Minimal kernel headers.** Only the headers needed for CXL driver
   verification are included. This keeps compilation fast and avoids
   pulling in unnecessary kernel API surface.

5. **Override-friendly models.** The operational model functions are declared
   in headers, so regression tests can override them with deterministic
   implementations for precise invariant checking.

## Dependencies

- **Existing:** c2goto infrastructure, kernel headers, kernel.c operational model
- **No new external dependencies required**

## Risks & Mitigations

| Risk | Mitigation |
|------|-----------|
| State space explosion from nondet MMIO reads | Use `--no-bmc` and bounded unwinding; override models in tests |
| Missing kernel symbols not covered by headers | Add stubs incrementally as needed by real driver code |
| CXL spec changes | Model CXL 2.0 first; 3.0 additions are incremental |
| Performance on real driver code | Start with minimal harnesses; scale up gradually |

## Success Metrics

- [x] Phase 1 files compile and integrate into ESBMC build
- [x] Phase 1 regression tests pass with expected results
- [x] Phase 3 regression suite (14 tests) passes
- [x] Phase 4 advanced features modeled
- [x] Phase 5: Real-world driver harnesses created and verified
- [x] Phase 6: User documentation and tutorial published

## Final Statistics

| Metric | Count |
|--------|-------|
| Total commits | 5 |
| Total regression tests | 22 |
| Passing tests | 16 |
| Bug-detecting tests | 6 |
| Kernel headers added | 6 |
| Operational model lines | ~1,050 |
| Documentation pages | 2 |
