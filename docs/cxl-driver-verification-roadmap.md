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
| `cxl_dvsec_rr_decode()` | `info->dvsec_range[2]` store stays in bounds (`ranges <= 2`) | SUCCESSFUL |
| `cxl_dvsec_rr_decode()` | liveness: asserting `ranges <= 1` | FAILED — both ranges reachable |
| `cxl_hdm_decode_init()` | no out-of-bounds read, caller contract assumed | SUCCESSFUL |
| `cxl_hdm_decode_init()` | same, contract dropped | FAILED at `range.h:20` |

Both bounded harnesses also verify with unwinding assertions enabled, so the
`--unwind` bounds are sound rather than truncated.

`cxl_dvsec_rr_decode()` and `cxl_hdm_decode_init()` are joined by a precondition
that appears nowhere in the code: the first caps `info->ranges` at 2 via its
`hdm_count > 2` rejection, and the second indexes `info->dvsec_range[i]` for
`i < info->ranges` without re-validating it. Safe today because `core/hdm.c`
calls them in sequence on the same `info`.

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

### Phase 7: Real Driver Coverage (Started)

**Goal:** Extend the operational model and regression suite to cover the full
Linux 7.1.5 CXL driver subsystem (~30 source files across 11 driver families),
moving from synthetic primitives to real-driver-equivalent verification.

**Status:** the first slice is in. `cxl_memdev_*` and `cxl_region_*` (tests 1-4
below) are implemented and passing, along with the five operational-model
functions they need. These are the first models written against the real
driver's constraints rather than the invented API in the scope note:
`cxl_memdev_id_alloc()` reproduces the `-ENOSPC` result of
`ida_alloc_range()`, and `cxl_region_config()` enforces the power-of-two
interleave encodings from `drivers/cxl/core/region.c`. Writing them exposed a
latent defect — the model's own allocations called a `static` `__kmalloc()`
through an implicit declaration, which no test had ever reached.

The second slice adds the mailbox IOCTL path (`core/mbox.c`) and downstream
port lifetime (`port.c`) — tests 7-10 — bringing the suite to 35. Test 8 was
retargeted: an unsupported opcode is already covered as a rejection assertion
in `cxl_mbox_ioctl_01`, so the failing case models the more interesting bug,
a user-controlled payload length bounded against the mailbox limit rather
than the driver's own staging buffer. `cxl_dport_walk()` was dropped in
favour of `cxl_dport_find()` plus `cxl_dport_count()`, which express the same
traversal without a callback.

Both slices were put through a verification-soundness review: every model
branch was shown independently reachable, and each failing test was
patched-and-reverified to confirm it flips to SUCCESSFUL once the modelled
driver bug is fixed, rather than failing for an incidental reason. That
review also found the missing overflow precondition on
`cxl_region_overlaps()`, since fixed.

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

**Real files no proposed test covers.** Checked against the Linux 7.1.5 tree,
not from memory — `drivers/cxl/core/` really does hold 20 `.c` files as the
table above says, but seven of them are claimed by no test row below:

| File | Why it matters |
|---|---|
| `core/regs.c` | Register block mapping — the foundation the whole MMIO model rests on |
| `core/atl.c` | Address translation; directly under the region/HPA work in tests 3-4 |
| `core/pmem.c` | Distinct from the top-level `pmem.c` that test 11 covers |
| `core/ras_rch.c` | RCH error handling, distinct from `pcie/aer_cxl_rch.c` |
| `core/pmu.c` | See below |
| `core/suspend.c` | Suspend/resume paths |
| `core/trace.c` | Tracepoints; likely out of scope, but should be said rather than omitted |

`core/pmu.c` and `drivers/perf/cxl_pmu.c` are **two different files** and both
exist. The family table lists "pmu" under CXL Core while test 21 targets
`perf/cxl_pmu.c`, so `core/pmu.c` currently has no owner.

**Proposed regression tests (~25 new):**

| # | Suite | Target | Feature | Expected |
|---|---|---|---|---|
| 1 | `cxl_memdev_01` | `core/memdev.c` | `/dev/cxl/X` creation + fw version | PASS — **done** |
| 2 | `cxl_memdev_02` | `core/memdev.c` | memdev ID allocator overflow | FAIL — **done** |
| 3 | `cxl_region_01` | `core/region.c` | Region interleave config | PASS — **done** |
| 4 | `cxl_region_02` | `core/region.c` | Overlapping region targets | FAIL — **done** |
| 5 | `cxl_region_dax_01` | `core/region_dax.c` | DAX region mapping | PASS |
| 6 | `cxl_region_pmem_01` | `core/region_pmem.c` | PMEM region type | PASS |
| 7 | `cxl_mbox_ioctl_01` | `core/mbox.c` | IOCTL command table lookup | PASS — **done** |
| 8 | `cxl_mbox_ioctl_02` | `core/mbox.c` | Unchecked payload size via IOCTL | FAIL — **done**, retargeted |
| 9 | `cxl_port_dport_01` | `port.c` | Downstream port traversal | PASS — **done** |
| 10 | `cxl_port_dport_02` | `port.c` | Dangling dport reference | FAIL — **done** |
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
| `cxl_memdev_create()`, `cxl_memdev_destroy()` | memdev lifecycle — **done** |
| `cxl_memdev_id_alloc()` | ID allocator with overflow — **done** |
| `cxl_region_config()` | Interleave granularity/size — **done** |
| `cxl_region_overlaps()` | HPA intersection test — **done**, not in the original plan |
| `cxl_dax_region_map()` | DAX region mapping |
| `cxl_pmem_region_type()` | PMEM region type check |
| `cxl_mailbox_ioctl()` | IOCTL command dispatch — **done** |
| `cxl_mbox_cmd_index()` | Command table lookup — **done**, not in the original plan |
| `cxl_dport_add()`, `cxl_dport_find()`, `cxl_dport_remove()`, `cxl_dport_count()` | Port downstream traversal — **done**; `_walk()` was replaced by find/count |
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

---

### Phase 8: Assurance (Started — should precede more breadth)

**Goal:** make the suite establish what it appears to establish. Phases 1-7
grew coverage outwards; this one closes the gap between the number of tests
and the amount of assurance they provide. Each item below comes from a
measured defect, not a stylistic preference.

**1. Declare the properties each test verifies.** — Done.

Every `test.desc` in this suite historically left the flags line empty, so all
of them ran on ESBMC defaults. `--memory-leak-check`, `--overflow-check`,
`--data-races-check` and `--deadlock-check` are opt-in and none were used.
This is not hypothetical: enabling `--memory-leak-check` immediately exposed
a real CWE-401 leak in `cxl_port_dport_01` on its bail-out path, in a test
that had been committed green the same day. A test that does not say which
properties it checks is not guarding them.

Every test now declares `--memory-leak-check --overflow-check
--unsigned-overflow-check`. All 35 pre-existing tests hold under them, so the
dport leak was the only latent bug of those classes, and the checks cost
nothing measurable — the suite runs in ~75s either way. Treat the flags line
as part of a test's specification in review from here on.

**2. Measure model coverage, not test count.** — Done.

Only 11 of the 35 tests ever call a modelled function without also defining
it locally. The other 24 declare their own structs and driver functions and
verify those, so deleting `cxl_driver.c` would leave most of the suite
passing. Several tests go further and *shadow* modelled functions with local
definitions — `cxl_mem_attach_01` and `cxl_port_enum_01` do this, as do the
four HDM-alignment tests.

The distribution is worse than the ratio suggests: 8 of those 11 were added
in the Phase 7 work, so for most of this project's life exactly **3 of 27**
tests exercised the model at all — the MMIO read-back pair and
`cxl_mmio_01`.

That is why `cxl_driver.c` called a `static __kmalloc()` through an implicit
declaration for its entire existence without any test noticing: **the model's
own allocation code had never once executed.**

`scripts/cxl_model_coverage.py` now reports this. The current figure is
**20 of 105 modelled functions called by any test — 19%** — against 13 of 37
tests that call into the model at all. Track that number, not the test count.
The script also reports GOTO linkage and documents why that figure
over-reports (33 of 37) and must not be used as the coverage number.

**3. Get the real-driver harnesses into CI.** — Partly done; the original
plan is ruled out.

`regression/cxl-linux/` holds the only work that touches real kernel source —
and it is unregistered with ctest because it needs a configured kernel tree.
The project's most valuable output is therefore its least protected, free to
rot silently.

This item originally proposed vendoring preprocessed translation units (`.i`
files) so Phase 5 would reproduce without a kernel checkout. That is not
viable, on three counts measured rather than guessed:

- **Size.** One preprocessed harness is 2.9 MB / 67,943 lines. Eight of those
  is a larger payload than the rest of `regression/` combined, for eight
  verified functions.
- **ESBMC cannot parse them.** Feeding the gcc-preprocessed `.i` back in gives
  `ERROR: PARSING ERROR` at `./drivers/cxl/core/pci.c:582`, on
  `} else if (((uport)->bus == &pci_bus_type)) {`. The vendored artefact would
  not even run.
- **Licence.** `drivers/cxl/core/pci.c` is `SPDX-License-Identifier:
  GPL-2.0-only`. ESBMC's `COPYING` already calls its licensing situation
  complex; copying kernel source into the tree makes that worse for no gain.

What is done instead: the harness verdicts are now executable rather than
prose. `regression/cxl-linux/run_all.sh` encodes the README's
harness→flags→expected-verdict table — eight verdict cases, an unwinding
soundness case (the bounded harnesses must still hold with unwinding
assertions *on*, or the bound is an artefact of truncation), and the
conversion check for `core/pci.c`. It exits non-zero on any mismatch, so the
table can no longer drift from reality unnoticed.

Making the table executable found two defects in it straight away, which is
the argument for the exercise:

- `run_esbmc.sh` `cd`s into the kernel tree before resolving the harness path,
  so **the README's own documented invocation could not work as written**.
  Fixed with `realpath`.
- The two HDM rows recorded no `--unwind` bound. Both harnesses cap
  `info.ranges` with `__ESBMC_assume()`, but an assumption constrains the
  value, not the unwinding — `cxl_hdm_decode_init()`'s loop over
  `info->ranges` (`pci.c:428`) still unwinds syntactically, reaching iteration
  4562 and exhausting 12 GB. As written the row was unreproducible. With
  `--unwind 3` / `--unwind 5` (one past each harness's assumed maximum, and
  with unwinding assertions **on**, so the bound is proved) both verdicts
  reproduce exactly as claimed, in under a second each.

The second is the more uncomfortable of the two: a verdict recorded without
the flags that produce it is not a result, and this one sat in the docs
unchallenged since Phase 5. `run_all.sh` also caps every run with
`--memlimit`/`--timeout`, after an unguarded run took this 30 GiB machine down
to 3 GiB free.

All ten checks now pass.

What remains: this still needs a kernel tree, so it is a local gate, not a CI
one. The viable CI route is a separate non-blocking workflow that prepares a
kernel tree, not vendoring.

**4. Verify concurrency at all.** — Done.

CXL exists to extend cache coherence across a link, and the Target Scope
lists concurrent driver access and DMA coherence. Yet `cxl_concurrent_01` —
the suite's only "concurrent" test — spawns no threads. It calls
`submit_command()` three times in sequence, where `command_count == 3` is
trivially true. No test in the suite explores an interleaving, and Phase 7
proposes no concurrency tests either.

`cxl_mbox_race_01` / `cxl_mbox_race_02` close the first half of this. Two
threads drive the single mailbox register set that `core/mbox.c` serialises on
`cxl_mailbox::mbox_mutex`; without the lock ESBMC reports a W/W data race on
the in-flight counter (CWE-362), and the passing variant is the same test with
the mutex taken — so the pair is its own patch-and-reverify. Both run under
`--data-races-check --context-bound 2`.

`cxl_concurrent_01` is now genuinely threaded — two submitters and an error
handler contending for the same device state, with a blocking spinlock.
Deleting its lock makes ESBMC report a W/W data race on the counter, so it
passes because the lock works rather than because nothing contends.

`cxl_dma_coherent_01/02` cover the DMA half. Coherent means the CPU and device
observe each other's writes without cache maintenance; it does not mean they
may write the same word at once, so ownership still has to be handed over. The
failing variant hands nothing over.

Five tests now verify an interleaving, all under `--data-races-check
--context-bound 2`.

**5. Require patch-and-reverify for every failing test.** — Done as practice.

A test expecting `VERIFICATION FAILED` can fail for a reason unrelated to the
bug it claims to model. The practice that works: patch the modelled bug in a
scratch copy and confirm the test flips to `VERIFICATION SUCCESSFUL`. This was
done for `cxl_memdev_02`, `cxl_region_02`, `cxl_mbox_ioctl_02` and
`cxl_port_dport_02`; make it the documented standard for new failing tests
rather than an occasional courtesy.

**6. Resolve the kernel-version incoherence.** — Done; it was not what this
item claimed.

This item asserted that the model's headers being under
`ubuntu20.04/kernel_5.15.0-76/` while real-driver work targets Linux 7.1.5
was an unresolved incoherence needing a decision on which is authoritative.
Checking it dissolved the question on three counts:

- **The two header sets never meet.** No file includes both:
  `regression/cxl-linux/` compiles against a real kernel tree's own headers,
  `regression/cxl/` against the model's. There is no mechanism by which they
  can drift into disagreement — the risk row that said otherwise was wrong.
- **The CXL headers are not a pinned copy of anything.** Linux has no
  `include/linux/cxl.h` in any version; the real declarations live in
  `drivers/cxl/cxl.h` under different names. `cxl.h` and `cxlmem.h` here are
  the synthetic API, in a directory named for a kernel they never came from.
- **The directory predates this work.** It was created by `b074f63dab`
  ("implement verificatin for kmalloc") and is shared with `kernel.c`. It is
  simply where ESBMC keeps kernel operational-model headers; renaming it
  would be an ESBMC-wide change, and a cosmetic one.

So there was no version to pick. What there was is a path making a claim that
is not true, now corrected where it can actually be read — in the header
comments of `cxl.h` and `cxlmem.h`.

The real question underneath is whether the model should mean the same thing
as the Linux CXL API at all (`struct cxl_dev` vs `struct cxl_memdev`,
`cxl_mailbox_send_cmd()` vs `cxl_internal_send_cmd()`). That is Phase 7's
convergence work and is tracked there, not here.

**7. Raise model coverage, and expect it to find defects.** — In progress.

Coverage went from 22 of 105 (21%) to 61 of 107 (57%) by writing five test
pairs against the surfaces nothing had ever called: PCI probe/teardown, PCI
config space, MSI/IRQ lifetime, streaming DMA, and AER. Reaching them for the
first time is what exposed the following, all of which had shipped green:

- **`pci_get_device()` divided by zero (CWE-369).** It fell back to
  `nondet % esbmc_pci_count` when the id did not match, and `esbmc_pci_count`
  was *never incremented anywhere* — the device table had no population API at
  all, so it was permanently 0. The fallback was also unsound in its own right:
  it returned a device that did not match the requested vendor/device id.
  Removed, and `esbmc_pci_register_device()` added so a harness can populate
  the table.
- **The entire AER layer was dead code.** `__esbmc_get_aer_cap()` looked state
  up by comparing `&esbmc_pci_devices[i] == dev`, but callers pass their own
  `struct pci_dev`, which is not in that table. The lookup could therefore
  never match: `pci_enable_aer()` leaked a fresh slot on every call, and
  `pci_aer_get_first_error()` / `pci_aer_clear_first_error()` always returned
  `-ENODEV`. All four AER functions were unusable.
- **Side-table lookups were unaffordable.** Keying AER state on a scan of a
  16-slot array cost a symbolic pointer comparison per slot per call, and a
  seven-call test did not solve in 300 s. AER state now lives in
  `struct pci_dev`, as it does in Linux; the same test solves in 0.14 s.
- **Loops bounded by mutable counters go symbolic.** `esbmc_simulate_irq()`
  and the old AER scan loop are bounded by a static counter, which stops being
  concrete as soon as a failure path merges — symex then unwinds them
  indefinitely. Tests touching them need an explicit `--unwind` (and the AER
  scan was rewritten to a compile-time bound). This is the same trap that made
  Phase 5's HDM harness unreproducible, in a second place.

Two of these mean a modelled function could not have worked for any caller.
That is the argument for coverage over test count, in one line.

**Acceptance criteria:**
- Every test declares its property flags; the leak and overflow classes are
  checked wherever they apply.
- A model-coverage figure exists and is reported alongside the test count, and
  is moving: 21% -> 57%.
- Phase 5's harness verdicts are checked by a script rather than asserted in
  prose (`run_all.sh`); running them in CI still needs a prepared kernel tree.
- At least two genuinely concurrent tests exist and pass under
  `--data-races-check`.
- No failing test is added without a recorded patch-and-reverify.


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
├── cxl_mmio_readback_02/             # Unwritten register is not fixed (FAIL)
├── cxl_memdev_01/                    # memdev create/destroy lifecycle (PASS)
├── cxl_memdev_02/                    # Unchecked memdev id allocation (FAIL)
├── cxl_region_01/                    # Region interleave validation (PASS)
├── cxl_region_02/                    # Overlapping region targets (FAIL)
├── cxl_mbox_ioctl_01/                # Mailbox IOCTL validation (PASS)
├── cxl_mbox_ioctl_02/                # Unchecked IOCTL payload size (FAIL)
├── cxl_port_dport_01/                # Downstream port traversal (PASS)
└── cxl_port_dport_02/                # Dangling dport reference (FAIL)
```

### Tooling

```
scripts/
└── cxl_model_coverage.py             # modelled functions exercised by tests
```

Run it with `--esbmc build/src/esbmc/esbmc` to include the GOTO-linkage
figure; without it only the static analysis runs.

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

   This one has a cost that went unrecorded for most of the project's life.
   Shadowing is what let most of the suite drift into verifying harnesses
   written inside the test file: `cxl_mem_attach_01` and `cxl_port_enum_01`
   both define local copies of functions the model provides, so the model's
   own code never ran. That is the direct cause of `cxl_driver.c` calling a
   `static __kmalloc()` through an implicit declaration undetected. Overriding
   remains useful for pinning a specific invariant, but a test that overrides
   is testing itself, and should not be counted as covering the model — which
   is what `scripts/cxl_model_coverage.py` now measures.

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
| Suite cost outgrowing its value | All cxl tests are THOROUGH and the suite is opt-in (`-DENABLE_CXL_REGRESSION=On`); one test accounts for most of the wall time |
| Model and real driver drifting apart | Not a live risk, and this row previously overstated it — no file includes both header sets, so they cannot disagree. What they *can* do is stay unrelated; closing that is Phase 7, not versioning. See Phase 8.6 |
| Tests that pass without establishing anything | Phase 8.1-8.2: declare property flags per test, measure model coverage rather than test count |

## Success Metrics

- [x] Phase 1 files compile and integrate into ESBMC build
- [x] Phase 1 regression tests pass with expected results
- [x] Phase 3 regression suite passes (10 new tests, 22 cumulative at that point)
- [x] Phase 4 advanced features — regression tests cover mailbox state,
        HDM validation (with alignment + decoder limit constraints),
        AER (with operational model functions), error injection (with
        operational model functions), and port enumeration
- [x] Phase 5: first real driver harnesses created and verified —
        `drivers/cxl/core/pci.c` converts; `cdat_checksum()`,
        `cxl_pci_get_latency()`, `cxl_dvsec_rr_decode()` and
        `cxl_hdm_decode_init()` verified (see Phase 5); broad coverage pending
- [x] Phase 6.1: User documentation published (user guide + roadmap + test summary)
- [ ] Phase 6.2–6.3: Generic driver template and technical report — not started
- [~] Phase 8: 8.1 (property flags), 8.2 (coverage metric), 8.4
        (concurrency), 8.5 (patch-and-reverify, by construction in the
        race pairs) and 8.6 (kernel version — dissolved, see the item) done;
        8.7 (coverage) in progress at 57%; 8.3 (real-driver harnesses in CI)
        blocked on a kernel tree, vendoring ruled out
- [~] Phase 7: two slices delivered — memdev id allocation, region
        interleave, the mailbox IOCTL path and downstream port lifetime,
        all modelled against the real driver's constraints (8 tests, 11
        model functions); remaining ~17 tests and ~9 functions pending

**On reading these numbers.** Test and line counts measure output, not
assurance, and this document has historically reported only those. Three
figures say more about what is actually established, and two of them are
currently poor:

| Question | Today | Was |
|---|---|---|
| Real Linux driver functions verified | 4 |
| Model functions exercised by tests | 61 of 107 (57%) | 4 |
| Operational model functions exercised | 61 of 107 (57%) | 22 of 105 (21%) |
| Tests that execute the operational model | 25 of 49 | 15 of 39 |
| Tests declaring the properties they check | 49 of 49 | 1 of 35 |
| Tests that verify an interleaving | 5 of 49 | 0 of 35 |

Phase 8.1 and 8.4 moved the last two; 8.2 made the second measurable for the
first time. Raising it from 21% to 57% is what found the model defects listed
under Phase 8.7 — the PCI and AER surfaces had never been executed, and three
of them did not work at all. A rising test count still should not be read as
rising confidence; a rising coverage figure is worth more, because reaching a
function is what exposes it.

## Current Statistics

**Suite status:** every cxl test is classified `THOROUGH`, and the suite is
**not registered by default** — `regression/CMakeLists.txt` gates it behind
`ENABLE_CXL_REGRESSION`, which is `OFF`. CI therefore does not run it. Run it
locally with:

```sh
cmake -DENABLE_CXL_REGRESSION=On -Bbuild -S . && ctest -L cxl
```

This keeps CI budget on suites that guard shipped behaviour, at the cost of
leaving the CXL work unguarded against regressions from elsewhere in the tree.
That trade-off is only sound while the suite stays largely self-contained; if
Phase 8.2 raises model coverage, the suite becomes worth re-enabling.


| Metric | Count |
|--------|-------|
| Total commits | 34 |
| Total regression tests | 49 |
| Passing tests | 29 |
| Bug-detecting tests | 20 |
| Kernel headers added | 6 |
| Operational model lines | 1611 |
| Documentation pages | 3 |
| AER functions added | 4 |
| Error injection functions added | 2 |
| HDM constraints added | 2 (alignment + decoder limit) |
| Real Linux driver files converted to GOTO | 1 |
| Real Linux driver functions verified | 4 |
| Model functions exercised by tests | 61 of 107 (57%) |
