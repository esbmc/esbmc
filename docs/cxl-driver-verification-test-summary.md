# CXL Regression Test Suite — Verification Strategy & Verdicts

## Flags & Configuration

All 27 tests share the same configuration:

| Property | Value |
|---|---|
| **test.desc mode** | `CORE` (no KNOWNBUG/FUTURE/THOROUGH tags) |
| **ESBMC mode** | BMC (default, with automatic unwinding) |
| **Additional flags** | None — line 3 of every `test.desc` is present but empty |
| **Test runner env** | `ESBMC_REGRESS_TIMEOUT=1200`, `ESBMC_REGRESS_MEMORY_LIMIT=8192` |
| **Expected verdicts** | `VERIFICATION SUCCESSFUL` (PASS tests) or `VERIFICATION FAILED` (FAIL tests) |

**Assessment of flags:** The tests are **intentionally lightweight** — no `--unwind` overrides, no floating-point checks, no pointer checks, no concurrency (`--shared-variables`). This is appropriate for the small synthetic test programs (53–155 lines each) which exercise control flow, not large unrolled loops or complex data-race scenarios. The tests that use nondeterminism (`cxl_driver_irq_01`, `cxl_device_init_01`, and the two `cxl_mmio_readback_*` suites) are bounded — `cxl_driver_irq_01`'s loop runs 10 iterations — so BMC's default unwinding suffices.

> The empty line 3 is load-bearing. It is the ESBMC argument list; if it is
> omitted, the expected-output regex slides onto it and is passed to ESBMC as
> command-line arguments. ESBMC then aborts with `failed to figure out type of
> file`, the test is left with no regex to match, and it passes without
> verifying anything. Every suite here previously had this defect except
> `cxl_dma_01`.

---

## Test-by-Test Summary

### Category 1: PCIe AER / Error Handling (4 tests)

| # | Suite | Lines | Strategy | Expected Verdict | What's Checked |
|---|---|---|---|---|---|
| 1 | `cxl_aer_01` | 129 | assert chain | **SUCCESSFUL** | AER init sets counters to 0; correctable/non-fatal return 0; fatal returns -1; error counts increment correctly |
| 2 | `cxl_driver_aer_fatal_01` | 125 | assert + __ESBMC_assert | **SUCCESSFUL** | Real-driver probe sequence: pci_enable_aer → check fatal → abort with -EIO; AER enabled invariant holds |
| 3 | `cxl_error_01` | 109 | assert chain | **SUCCESSFUL** | Error injection: correctable (log+continue), non-fatal (reset+recover), fatal (system_dead=1, return -1); counts verified |
| 4 | `cxl_irq_02` | 67 | __ESBMC_assert invariant | **FAILED** (bug) | Double-free: free_irq called twice, free_count=2, invariant `free_count <= 1` violated |

### Category 2: HDM Decoder Management (4 tests)

| # | Suite | Lines | Strategy | Expected Verdict | What's Checked |
|---|---|---|---|---|---|
| 5 | `cxl_hdm_01` | 81 | assert chain | **SUCCESSFUL** | Correct overlap check: 3 non-overlapping 256MB decoders; decoder 0 limit < decoder 2 base |
| 6 | `cxl_hdm_overlap_01` | 91 | __ESBMC_assert invariant (nested loop) | **FAILED** (bug) | Missing overlap check: decoders 0 (0–256MB) and 1 (128–384MB) overlap, invariant violated |
| 7 | `cxl_driver_hdm_align_01` | 83 | assert chain | **SUCCESSFUL** | Alignment enforcement: 4 aligned bases (0, 256MB, 512MB, 1GB) all accepted; decoder_count ≤ 8 |
| 8 | `cxl_driver_hdm_align_fail_01` | 68 | __ESBMC_assert invariant | **FAILED** (bug) | Missing alignment check: base 0x80000080 not 4KB-aligned, invariant `(addr % 4096) == 0` violated |

### Category 3: Driver Lifecycle / Probe & Remove (2 tests)

| # | Suite | Lines | Strategy | Expected Verdict | What's Checked |
|---|---|---|---|---|---|
| 9 | `cxl_driver_probe_01` | 155 | assert chain | **SUCCESSFUL** | Full probe: enable → request_regions → iomap → device_init; remove: iounmap → release → disable; driver_data NULL after remove |
| 10 | `cxl_driver_remove_01` | 101 | __ESBMC_assert invariant | **FAILED** (bug) | Missing free_irq in remove: irq_registered stays 1, invariant `irq_registered == 0` violated |

### Category 4: Mailbox / Command Queue (2 tests)

| # | Suite | Lines | Strategy | Expected Verdict | What's Checked |
|---|---|---|---|---|---|
| 11 | `cxl_mailbox_01` | 77 | assert (direct) | **FAILED** (bug) | Missing return value check: cxl_mailbox_send_cmd returns -EIO (status=1), but driver asserts `cmd.status == 0` without checking the return |
| 12 | `cxl_mailbox_state_01` | 117 | assert chain | **SUCCESSFUL** | State machine: IDLE → BUSY → COMPLETE → IDLE cycle (2 commands, one success + one error status); asserts block each state transition |

### Category 5: Security State Machine (2 tests)

| # | Suite | Lines | Strategy | Expected Verdict | What's Checked |
|---|---|---|---|---|---|
| 13 | `cxl_security_01` | 103 | assert chain | **SUCCESSFUL** | Valid transitions: NONE → PASSPHRASE_SET → UNLOCKED → LOCKED → DISABLED → NONE (full CXL 2.0 spec cycle) |
| 14 | `cxl_security_02` | 55 | __ESBMC_assert invariant | **FAILED** (bug) | Invalid transition: driver allows NONE → UNLOCKED (must go through PASSPHRASE_SET), state invariant violated |

### Category 6: Memory / Lifecycle / Partition (3 tests)

| # | Suite | Lines | Strategy | Expected Verdict | What's Checked |
|---|---|---|---|---|---|
| 15 | `cxl_mem_attach_01` | 88 | assert chain | **SUCCESSFUL** | Lifecycle: attach → enable → disable → detach; asserts enforce must-attached before enable, must-enabled before detach |
| 16 | `cxl_partition_01` | 77 | assert chain | **SUCCESSFUL** | Partition state machine: UNPARTITIONED → SPLIT (512/512MB) → SPLIT (256/768MB); total_size invariant preserved |
| 17 | `cxl_mmio_01` | 53 | wmb/mb no-op + trivial assert | **SUCCESSFUL** | Minimal smoke test: wmb() and mb() compile and run (no-op in model); null MMIO pointer handled gracefully |
| 17a | `cxl_mmio_readback_01` | 37 | assert chain over the model | **SUCCESSFUL** | Register read-back: write→read round-trip at 8/32/64-bit widths, later write wins, adjacent registers do not alias |
| 17b | `cxl_mmio_readback_02` | 25 | __ESBMC_assert invariant | **FAILED** (bug) | Guards against an over-constrained model: a register the driver never wrote must not read as a fixed value |

### Category 7: DMA (1 test)

| # | Suite | Lines | Strategy | Expected Verdict | What's Checked |
|---|---|---|---|---|---|
| 18 | `cxl_dma_01` | 133 | __ESBMC_assume precondition + assert | **FAILED** (bug) | DMA coherence: device writes 0x42 to buffer, driver reads without calling dma_sync_single_for_cpu first; assert `sync_called==1 || buffer==0` violated |

### Category 8: PCI Enumeration (2 tests)

| # | Suite | Lines | Strategy | Expected Verdict | What's Checked |
|---|---|---|---|---|---|
| 19 | `cxl_pci_enum_01` | 95 | assert chain | **SUCCESSFUL** | BAR mapping: find device (0x1234:0x0001) → read BAR0 start/end → iomap returns correct address |
| 20 | `cxl_pci_enum_02` | 57 | assert in caller | **FAILED** (bug) | NULL pointer dereference: pci_get_device returns NULL, driver calls pci_resource_start(dev, 0) without NULL check, assert inside fails |

### Category 9: Port Enumeration (1 test)

| # | Suite | Lines | Strategy | Expected Verdict | What's Checked |
|---|---|---|---|---|---|
| 21 | `cxl_port_enum_01` | 125 | assert chain | **SUCCESSFUL** | Port hierarchy: create root port (type=ROOT, rtype=CXL) → create 3 switch ports (type=DOWNSTREAM, parent=root); parent-of-parent == NULL invariant |

### Category 10: IRQ (2 tests)

| # | Suite | Lines | Strategy | Expected Verdict | What's Checked |
|---|---|---|---|---|---|
| 22 | `cxl_irq_01` | 97 | assert chain | **SUCCESSFUL** | IRQ lifecycle: request_irq → simulate 2 firings (counter=2) → free_irq → simulate (counter stays 2); handler=NULL after free |
| 23 | `cxl_driver_irq_01` | 93 | assert chain + nondet | **SUCCESSFUL** | IRQ dispatch: 10 iterations, type = nondet_int() % 3; each returns IRQ_HANDLED; total = mailbox + error + port counts |

### Category 11: Device Init (1 test)

| # | Suite | Lines | Strategy | Expected Verdict | What's Checked |
|---|---|---|---|---|---|
| 24 | `cxl_device_init_01` | 72 | __ESBMC_assume + __ESBMC_assert | **SUCCESSFUL** | Init sequence: read DEV_CTRL (0) → write INIT → assume DEV_STAT has INIT_DONE → write clear INIT + set ENABLE → verify ENABLE set, INIT cleared |

### Category 12: Concurrent / Atomic (1 test)

| # | Suite | Lines | Strategy | Expected Verdict | What's Checked |
|---|---|---|---|---|---|
| 25 | `cxl_concurrent_01` | 89 | assert chain | **SUCCESSFUL** | Spinlock data-race freedom: 3 concurrent command submits + 1 error handler; `__ESBMC_atomic_begin/end` blocks enforce mutual exclusion; command_count==3, error_count==1, lock==false |

---

## Verdict Summary

| Verdict | Count | Tests | Pattern |
|---|---|---|---|
| **VERIFICATION SUCCESSFUL** | **18** | see list below | Correct driver behavior — all assertions hold, invariants preserved |
| **VERIFICATION FAILED** | **9** | see list below | Intentional bugs — driver violates spec/contract |

**SUCCESSFUL (18):** `cxl_aer_01`, `cxl_driver_aer_fatal_01`, `cxl_error_01`,
`cxl_hdm_01`, `cxl_driver_hdm_align_01`, `cxl_driver_probe_01`,
`cxl_mailbox_state_01`, `cxl_security_01`, `cxl_mem_attach_01`,
`cxl_partition_01`, `cxl_mmio_01`, `cxl_mmio_readback_01`, `cxl_pci_enum_01`,
`cxl_port_enum_01`, `cxl_irq_01`, `cxl_driver_irq_01`, `cxl_device_init_01`,
`cxl_concurrent_01`

**FAILED (9):** `cxl_irq_02` (double-free), `cxl_hdm_overlap_01` (missing
overlap check), `cxl_driver_hdm_align_fail_01` (missing alignment check),
`cxl_driver_remove_01` (missing IRQ cleanup), `cxl_mailbox_01` (unchecked
return value), `cxl_security_02` (invalid state transition), `cxl_dma_01`
(missing DMA sync), `cxl_pci_enum_02` (NULL pointer dereference),
`cxl_mmio_readback_02` (unwritten register assumed fixed)

18 + 9 = 27, matching `ctest -L cxl`.

---

## Verification Strategy Assessment

### What Works Well

1. **assert vs __ESBMC_assert distinction is correct.**
   - PASS tests use `assert()` to verify happy-path control flow (execution never reaches a violated assertion).
   - FAIL tests use `__ESBMC_assert()` for invariant checking (ESBMC proves these can be violated).

2. **__ESBMC_assume used correctly** in device_init_01 (to model polling loop convergence) and dma_01 (to set up the precondition that device wrote data to the buffer).

3. **Bug injection patterns are realistic** — they match common driver bugs:
   - Missing NULL check (pci_enum_02)
   - Missing return value check (mailbox_01)
   - Missing resource cleanup (remove_01)
   - Missing synchronization (dma_01)
   - Missing validation (hdm_align_fail_01, hdm_overlap_01)
   - Missing state machine enforcement (security_02, irq_02)

4. **cxl_driver_* tests mirror the *shape* of real driver code** — probe/remove ordering, IRQ cleanup, AER abort paths. They do not compile or link any kernel source, and the CXL API they call is synthetic (see the scope note in the roadmap); `drivers/cxl/cxl_core.c` does not exist in the kernel.

### Recommendations

1. **Roadmap statistics** are now 18 passing / 9 bug-detecting across 27 suites, matching `ctest -L cxl`.

2. **Flags are adequate but sparse.** No test uses:
   - `--z3` / `--bitwuzla` solver selection — all tests run with the default SMT solver
   - `--no-bmc` — not needed (BMC is the default and correct mode for these tests)
   - `--floats` — no floating-point code in any test
   - `--pointer-check` — pointer bugs are tested via `assert` (e.g., pci_enum_02), not the built-in pointer checker. For deeper pointer analysis (buffer overflows, OOB), this could be added.
   - `--shared-variables` — concurrent_01 uses `__ESBMC_atomic_begin/end` instead of shared-variable interleaving. Both are valid; the atomic approach is simpler and sufficient for these small programs.

3. **cxl_driver_irq_01** uses `__VERIFIER_nondet_int() % 3` to explore all 3 IRQ types over a 10-iteration loop — small enough that BMC unwinding won't explode. `cxl_irq_01`, despite the similar name, is fully deterministic and has no loop.

4. **No tests use `--k-induction` or `--unwind N`.** For the small programs here, default BMC with automatic unwinding is appropriate. If tests grow to include unbounded loops, k-induction would become necessary.
