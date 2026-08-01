# CXL Driver Verification Roadmap for ESBMC

## Overview

This document outlines the phased roadmap for adding CXL (Compute Express Link)
driver verification support to ESBMC. CXL is an industry-standard chip-to-chip
interconnect that extends CPU memory and I/O coherence to accelerators, storage,
and other devices. Verifying CXL drivers is critical because bugs in these
kernel-mode drivers can cause system crashes, data corruption, and security
vulnerabilities.

> **Scope note — the modelled API is synthetic.** The operational model in
> `src/c2goto/library/cxl_driver.c` implements a *CXL-like* API invented for
> this work, not the Linux CXL API. Names such as `struct cxl_dev`,
> `cxl_mailbox_send_cmd()`, `cxl_device_init()`, `cxl_setup_hdm_decoders()` and
> `pci_enable_aer()` do not exist in Linux 7.1.5 (the real equivalents are
> `struct cxl_memdev` / `struct cxl_dev_state`, `cxl_internal_send_cmd()`,
> `pci_aer_clear_nonfatal_status()`, …). The generic kernel primitives the model
> also provides — `readl`/`writel`, `pci_iomap`, `dma_alloc_coherent`,
> `request_irq` — *are* real. Every regression test is therefore a synthetic
> harness exercising driver *patterns*; none compiles real kernel source.
> Closing that gap is the subject of Phase 7.

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

**Implemented additions:**

1. **Mailbox protocol state machine.**
   - Synthetic regression test (`cxl_mailbox_state_01`) models the mailbox
     command state machine with polling for completion.
   - Does **not** model hardware busy bits or command-completion interrupts
     in the operational model.

2. **CXL port and switch enumeration.**
   - Synthetic regression test (`cxl_port_enum_01`) walks a simulated port
     hierarchy.
   - Does **not** model ACPI _CCA, _CRS, or _DSM methods.

3. **HDM (Host Memory Decode) decoder validation.**
   - Two regression tests: `cxl_hdm_01` (valid setup) and
     `cxl_hdm_overlap_01` (overlapping regions — detected as a bug).
   - Operational model now enforces 4KB alignment on base addresses and the
     8-decoder limit per CXL 2.0 §8.2.2.12.
   - Two additional regression tests:
     `cxl_driver_hdm_align_01` (aligned addresses succeed) and
     `cxl_driver_hdm_align_fail_01` (misaligned addresses rejected).

4. **CXL error handling.**
   - Synthetic regression test (`cxl_error_01`) exercises driver error
     injection paths.
   - Operational model now provides `cxl_err_inject()` and
     `cxl_err_get_count()` for error injection and counting.

5. **PCIe AER (Advanced Error Reporting).**
   - Synthetic regression test (`cxl_aer_01`) exercises error recovery paths.
   - Operational model now provides `pci_enable_aer()`, `pci_aer_clear()`,
     `pci_aer_get_first_error()`, and `pci_aer_clear_first_error()`.

---

### Phase 5: Real-World Validation (Started)

**Goal:** Apply the CXL verification infrastructure to real Linux CXL driver
code and validate against known bugs.

**Status:** `drivers/cxl/core/pci.c` from Linux 7.1.5 now converts to a GOTO
program (5233 functions) and two of its functions are verified. Harnesses,
prerequisites and flags are in `regression/cxl-linux/`; they need a configured
kernel tree, so they are not registered with ctest and do not run in CI.

| Function | Property | Result |
|---|---|---|
| `cdat_checksum()` | no out-of-bounds read for `size <= sizeof(buf)` | SUCCESSFUL |
| `cdat_checksum()` | negative variant, bound relaxed | FAILED at `pci.c:554` |
| `cxl_pci_get_latency()` | division by zero, callee unconstrained | FAILED at `pci.c:667` |
| `cxl_pci_get_latency()` | same, callee contract modelled | SUCCESSFUL |

The `cxl_pci_get_latency()` failure is a false positive, not a kernel bug: the
`bw < 0` guard is sufficient only because `pcie_dev_speed_mbps()` returns
`-EINVAL` or a speed `>= 2500`. That invariant is unstated in the code, which
is the general lesson — verifying against undefined kernel functions requires
modelling each callee's contract, and each assumption is a soundness obligation.

Reaching this phase required four ESBMC fixes, none CXL-specific: the sign of
`void *` pointer subtraction (`value_set.cpp`), `offsetof` as a constant
expression, the unimplemented `--fms-extensions` option, `FileScopeAsm`
(`EXPORT_SYMBOL()`), and an anonymous-record tag collision that sent padding
computation into unbounded recursion on `__DECLARE_FLEX_ARRAY()`.

The `cxl_driver_*` suites remain synthetic harnesses that imitate driver
*shapes* against the invented API described in the scope note above; they still
establish nothing about the Linux CXL driver. Broad real-driver coverage is
Phase 7.

**Tasks:**

1. **Select target drivers.** (Paths verified against Linux 7.1.5.)
   - `drivers/cxl/core/port.c` — CXL core port/bus infrastructure
   - `drivers/cxl/pci.c` — CXL PCI device probe
   - `drivers/cxl/mem.c` — CXL memory device driver
   - `drivers/cxl/core/pci.c` — CXL PCI DVSEC/register setup

   Earlier revisions of this roadmap listed `drivers/cxl/cxl_core.c` and
   `drivers/cxl/pci_cxl.c`; neither file exists in the kernel tree.

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
   - Nothing has been reported or submitted upstream to date.

---

### Phase 6: Generalization & Documentation (Partially completed)

**Goal:** Make CXL verification accessible to other users and generalize
patterns for other device driver families.

**Tasks:**

1. **Write user documentation.** — Done.
   - How to write CXL driver verification tests.
   - How the operational models work.
   - How to extend the models for new CXL features.
   - User guide: `docs/cxl-driver-verification-guide.md`.

2. **Generalize patterns for other drivers.** — Not started.
   - The MMIO, DMA, and IRQ modeling patterns are generic.
   - Create a template for NVMe, USB, and other PCIe driver verification.

3. **Publish a technical report.** — Not started.
   - Document the methodology and findings.
   - Target a verification conference (CAV, TACAS, etc.).

---

## File Inventory

### Headers (Phase 1 + Phase 4 updates)

```
src/c2goto/headers/ubuntu20.04/kernel_5.15.0-76/include/
├── asm/
│   └── io.h                          # MMIO access functions
└── linux/
    ├── cxl.h                         # CXL core device API + AER + error injection
    ├── cxlmem.h                      # CXL memory device API
    ├── pci.h                         # PCI subsystem API
    ├── irq.h                         # Interrupt handling API
    └── dma-mapping.h                 # DMA API
```

`gfp.h`, `slab.h`, `spinlock.h` and `asm/uaccess.h` are also used by the model
but already existed on `master`; only the six files above were added here.

### Operational Model (Phase 1 + Phase 4 updates)

```
src/c2goto/library/
└── cxl_driver.c                      # CXL driver operational model
    ├── MMIO (readb/writel, barriers, block ops)
    ├── PCI (enumeration, BAR, MSI, driver registration)
    ├── AER (enable, clear, get/clear first error)
    ├── IRQ (request, free, simulate)
    ├── DMA (coherent alloc, streaming map, sync)
    ├── CXL core (enumerate, init, mailbox, security)
    ├── Error injection (inject, get counts)
    ├── HDM decoder (setup with alignment + 8-decoder constraints)
    └── CXL memory (attach, detach, regions, partition)
```

### Regression Tests (all phases)

```
regression/cxl/
├── cxl_mmio_01/                      # MMIO read/write correctness (PASS)
├── cxl_device_init_01/               # Device init sequence (PASS)
├── cxl_mailbox_01/                   # Mailbox status check (FAIL)
├── cxl_dma_01/                       # DMA sync correctness (FAIL)
├── cxl_irq_01/                       # IRQ registration/firing (PASS)
├── cxl_irq_02/                       # IRQ double-free (FAIL)
├── cxl_pci_enum_01/                  # PCI enumeration (PASS)
├── cxl_pci_enum_02/                  # NULL BAR access (FAIL)
├── cxl_partition_01/                 # Partition state machine (PASS)
├── cxl_security_01/                  # Valid security transitions (PASS)
├── cxl_security_02/                  # Invalid security transition (FAIL)
├── cxl_hdm_01/                       # HDM decoder setup (PASS)
├── cxl_hdm_overlap_01/               # Overlapping HDM decoders (FAIL)
├── cxl_concurrent_01/                # Concurrent access with spinlocks (PASS)
├── cxl_mem_attach_01/                # Memory device lifecycle (PASS)
├── cxl_aer_01/                       # PCIe AER error handling (PASS)
├── cxl_driver_irq_01/                # IRQ handler dispatch (PASS)
├── cxl_driver_probe_01/              # Full probe sequence (PASS)
├── cxl_driver_remove_01/             # Missing IRQ cleanup (FAIL)
├── cxl_error_01/                     # CXL error injection (PASS)
├── cxl_mailbox_state_01/             # Mailbox state machine (PASS)
├── cxl_port_enum_01/                 # Port hierarchy enumeration (PASS)
├── cxl_driver_hdm_align_01/          # HDM 4KB alignment validation (PASS)
├── cxl_driver_hdm_align_fail_01/     # HDM misaligned address rejection (FAIL)
├── cxl_driver_aer_fatal_01/          # AER fatal error during probe (PASS)
├── cxl_mmio_readback_01/             # MMIO write-then-read round-trip (PASS)
└── cxl_mmio_readback_02/             # Unwritten register is not fixed (FAIL)
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
- [x] Phase 3 regression suite passes (10 new tests, 22 cumulative at that point)
- [x] Phase 4 advanced features — regression tests cover mailbox state,
        HDM validation (with alignment + decoder limit constraints),
        AER (with operational model functions), error injection (with
        operational model functions), and port enumeration
- [x] Phase 5: first real driver harnesses created and verified —
        `drivers/cxl/core/pci.c` converts, `cdat_checksum()` and
        `cxl_pci_get_latency()` verified (see Phase 5); broad coverage pending
- [x] Phase 6.1: User documentation published (user guide + roadmap + test summary)
- [ ] Phase 6.2–6.3: Generic driver template and technical report — not started

## Current Statistics

| Metric | Count |
|--------|-------|
| Total commits | 10 |
| Total regression tests | 27 |
| Passing tests | 18 |
| Bug-detecting tests | 9 |
| Kernel headers added | 6 |
| Operational model lines | ~1,300 |
| Documentation pages | 3 |
| AER functions added | 4 |
| Error injection functions added | 2 |
| HDM constraints added | 2 (alignment + decoder limit) |
| Real Linux driver files converted to GOTO | 1 |
| Real Linux driver functions verified | 2 |

---

### Phase 7: Real Driver Coverage (Planned)

**Goal:** Extend the operational model and regression suite to cover the full
Linux 7.1.5 CXL driver subsystem (~30 source files across 11 driver families),
moving from synthetic primitives to real-driver-equivalent verification.

**Gap Analysis — Linux 7.1.5 `drivers/cxl/` inventory vs current coverage:**

| Driver Family | Kconfig | Source Files | Current Coverage |
|---|---|---|---|
| **CXL Core** (port, memdev, mbox, hdm, region, regs, cdat, features, ras, mce, pmu, suspend) | CXL_BUS | 20 `.c` files in `core/` | Partial — only synthetic port/mbox/region tests |
| **CXL PCI** (pci.c) | CXL_PCI | 1 | Partial — synthetic probe test, no DVSEC/doorbell/MMR timeout paths |
| **CXL Memory** (mem.c) | CXL_MEM | 1 | Partial — synthetic attach test, no endpoint enumeration |
| **CXL Port** (port.c) | CXL_PORT | 1 | Partial — synthetic enum test, no dport HDM scan |
| **CXL PMEM** (pmem.c + security.c) | CXL_PMEM | 2 | Partial — synthetic security tests, no LIBNVDIMM bridge |
| **CXL ACPI** (acpi.c) | CXL_ACPI | 1 | **None** — CEDT/CFMWS parsing untested |
| **DAX CXL** (dax/cxl.c) | DEV_DAX_CXL | 1 | **None** — DAX region probing untested |
| **CXL PMU** (perf/cxl_pmu.c) | CXL_PMU | 1 | **None** — perf event interface untested |
| **PCIe AER for CXL** (pcie/aer_cxl_rch.c) | built-in | 1 | Partial — basic AER covered, not RCH delegation |
| **ACPI CPER** (firmware/efi/cper_cxl.c) | built-in | 1 | **None** — CXL error section parsing untested |
| **ACPI EINJ** (acpi/apei/einj-cxl.c) | EINJ_CXL | 1 | **None** — ACPI error injection untested |

**Proposed regression tests (~25 new):**

| # | Suite | Target | Feature | Expected |
|---|---|---|---|---|
| 1 | `cxl_memdev_01` | `core/memdev.c` | `/dev/cxl/X` creation + fw version | PASS |
| 2 | `cxl_memdev_02` | `core/memdev.c` | memdev ID allocator overflow | FAIL |
| 3 | `cxl_region_01` | `core/region.c` | Region interleave config | PASS |
| 4 | `cxl_region_02` | `core/region.c` | Overlapping region targets | FAIL |
| 5 | `cxl_region_dax_01` | `core/region_dax.c` | DAX region mapping | PASS |
| 6 | `cxl_region_pmem_01` | `core/region_pmem.c` | PMEM region type | PASS |
| 7 | `cxl_mbox_ioctl_01` | `core/mbox.c` | IOCTL command table lookup | PASS |
| 8 | `cxl_mbox_ioctl_02` | `core/mbox.c` | Unsupported opcode via IOCTL | FAIL |
| 9 | `cxl_port_dport_01` | `port.c` | Downstream port traversal | PASS |
| 10 | `cxl_port_dport_02` | `port.c` | Dangling dport reference | FAIL |
| 11 | `cxl_pmem_sec_01` | `pmem.c` + `security.c` | Set/get passphrase flow | PASS |
| 12 | `cxl_pmem_sec_02` | `pmem.c` + `security.c` | Unlock before freeze (invalid order) | FAIL |
| 13 | `cxl_acpi_cedt_01` | `acpi.c` | CEDT CFMWS window parsing | PASS |
| 14 | `cxl_acpi_cedt_02` | `acpi.c` | CFMWS alignment violation | FAIL |
| 15 | `cxl_cdat_01` | `core/cdat.c` | CDAT latency/bandwidth parsing | PASS |
| 16 | `cxl_edac_01` | `core/edac.c` | Patrol scrub enable/disable | PASS |
| 17 | `cxl_features_01` | `core/features.c` | Feature capability query | PASS |
| 18 | `cxl_ras_01` | `core/ras.c` | CPER error record processing | PASS |
| 19 | `cxl_ras_02` | `core/ras.c` | CPER uncorrectable fatal w/o recovery | FAIL |
| 20 | `cxl_mce_01` | `core/mce.c` | MCE offlines aliased SPA page | PASS |
| 21 | `cxl_pmu_01` | `perf/cxl_pmu.c` | PMU counter configuration | PASS |
| 22 | `cxl_dax_01` | `dax/cxl.c` | DAX region device registration | PASS |
| 23 | `cxl_dax_02` | `dax/cxl.c` | DAX on non-DAX region type | FAIL |
| 24 | `cxl_einj_01` | `acpi/apei/einj-cxl.c` | EINJ error injection to CXL port | PASS |
| 25 | `cxl_dvsec_01` | `core/pci.c` | PCIe DVSEC register enumeration | PASS |

**New operational model functions (~18):**

| Function | Target |
|---|---|
| `cxl_memdev_create()`, `cxl_memdev_destroy()` | memdev lifecycle |
| `cxl_memdev_id_alloc()` | ID allocator with overflow |
| `cxl_region_config()` | Interleave granularity/size |
| `cxl_dax_region_map()` | DAX region mapping |
| `cxl_pmem_region_type()` | PMEM region type check |
| `cxl_mailbox_ioctl()` | IOCTL command dispatch |
| `cxl_dport_add()`, `cxl_dport_walk()` | Port downstream traversal |
| `cxl_pmem_set_passphrase()`, `cxl_pmem_unlock()` | LIBNVDIMM passphrase bridge |
| `acpi_cedt_parse_cfmws()` | CFMWS window parsing |
| `cdat_parse_entry()` | CDAT latency/bandwidth parsing |
| `cxl_edac_set_patrol_scrub()` | EDAC patrol scrub control |
| `cxl_feature_query()` | Feature capability table lookup |
| `cper_process_cxl()` | CXL CPER error record |
| `cxl_mce_offline_page()` | MCE SPA offline |
| `cxl_pmu_config_counter()` | PMU hardware config |
| `cxl_dax_region_register()` | DAX device registration |
| `einj_inject_cxl_error()` | ACPI EINJ error injection |
| `pci_cxl_dvsec_enum()` | DVSEC discovery |

**Projection after Phase 7:**

| Metric | After Phase 7 |
|---|---|
| Regression tests | ~50 |
| Passing tests | ~35 |
| Bug-detecting tests | ~15 |
| Operational model functions | ~70 |
| Operational model lines | ~3,000 |
| Driver families covered | ~9 |
