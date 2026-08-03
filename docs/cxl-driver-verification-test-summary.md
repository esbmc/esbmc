# CXL Regression Test Suite — Verification Strategy & Verdicts

## Flags & Configuration

All 71 tests share the same configuration:

| Property | Value |
|---|---|
| **test.desc mode** | `THOROUGH` — the suite is opt-in and CI does not run it (`-DENABLE_CXL_REGRESSION=On`) |
| **ESBMC mode** | BMC (default, with automatic unwinding) |
| **Additional flags** | `--memory-leak-check --overflow-check --unsigned-overflow-check`; the five concurrency tests use `--data-races-check --context-bound 2 --cswitch-skip-readonly-globals`; eighteen tests add an explicit `--unwind` (see below) |
| **Test runner env** | `ESBMC_REGRESS_TIMEOUT=1200`, `ESBMC_REGRESS_MEMORY_LIMIT=8192` |
| **Expected verdicts** | `VERIFICATION SUCCESSFUL` (PASS tests) or `VERIFICATION FAILED` (FAIL tests) |

**Assessment of flags.** An earlier revision of this document argued the tests were "intentionally lightweight" and that leaving line 3 empty was appropriate for small synthetic programs. That was wrong, and measurably so: the first time `--memory-leak-check` was enabled across the suite it found a CWE-401 leak in `cxl_port_dport_01`, in a test committed green the same day. A test that does not declare the properties it checks is not guarding them.

Every test now declares its flags, and all of them hold under leak and overflow checking. The cost is nil — the suite runs in ~75s either way, dominated by `cxl_driver_irq_01`.

Unwinding is left to BMC's default where the loops are concretely bounded (`cxl_driver_irq_01`'s runs 10 iterations, `cxl_device_init_01` and the `cxl_mmio_readback_*` pair likewise). Eighteen tests must state an `--unwind` instead, and the reason is worth knowing: a loop whose bound is a *mutable counter* in the model — `esbmc_irq_count`, or a device-supplied register value — stops being concrete as soon as a failure path merges, and symex then unwinds it indefinitely rather than solving the guard. `cxl_pci_config_01/02` use `--unwind 4`, `cxl_irq_msi_01/02` use `--unwind 2`, and `cxl_port_walk_01/02` use `--unwind 3`. Unwinding assertions are left **on** throughout, so every bound is proved rather than assumed.

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
| 25 | `cxl_concurrent_01` | 103 | threads + spinlock | **SUCCESSFUL** | Two submitter threads and an error handler contend for the same device state under a blocking spinlock; no update is lost and no access races. Mutation-checked: deleting the lock produces a W/W race on `command_count` |

Until Phase 8 this test spawned no threads at all — it called `submit_command()` three times in sequence, where `command_count == 3` holds trivially, and its `spin_lock()` was a try-lock whose failure the test asserted away (sound only when nothing contends, which is what the test claimed to rule out).

### Category 13: Model-facing lifecycle & geometry (8 tests)

These call the operational model rather than harnesses defined in the test file.

| # | Suite | Strategy | Expected Verdict | What's Checked |
|---|---|---|---|---|
| 26 | `cxl_memdev_01` | assert chain | **SUCCESSFUL** | `/dev/cxl/memN` lifecycle; id in range, fw_rev NUL-terminated |
| 27 | `cxl_memdev_02` | bounds | **FAILED** | Unchecked `ida_alloc_range()` result used as a table index |
| 28 | `cxl_region_01` | assert chain | **SUCCESSFUL** | Interleave encodings; each rejection branch independently reachable |
| 29 | `cxl_region_02` | `__ESBMC_assert` | **FAILED** | Second region committed without an overlap check |
| 30 | `cxl_mbox_ioctl_01` | assert chain | **SUCCESSFUL** | Opcode table lookup, payload bound, disabled-command rejection |
| 31 | `cxl_mbox_ioctl_02` | bounds | **FAILED** | User length bounded against the mailbox limit, not the staging buffer |
| 32 | `cxl_port_dport_01` | assert chain | **SUCCESSFUL** | dport register/find/remove; leak-checked on the bail-out path |
| 33 | `cxl_port_dport_02` | use-after-free | **FAILED** | dport pointer cached across the port's removal |

### Category 14: Concurrency (4 tests)

| # | Suite | Strategy | Expected Verdict | What's Checked |
|---|---|---|---|---|
| 34 | `cxl_mbox_race_01` | threads + mutex | **SUCCESSFUL** | Mailbox submission serialised on the equivalent of `cxl_mailbox::mbox_mutex` |
| 35 | `cxl_mbox_race_02` | threads, no lock | **FAILED** | W/W data race on the in-flight counter (CWE-362) |
| 36 | `cxl_dma_coherent_01` | threads + handover | **SUCCESSFUL** | Coherent buffer with explicit ownership handover; device observes the CPU's write |
| 37 | `cxl_dma_coherent_02` | threads, no handover | **FAILED** | W/W data race on the DMA buffer |

Each pair is a patched and unpatched version of the same program, so it carries
its own patch-and-reverify.

---

## Verdict Summary

| Verdict | Count | Tests | Pattern |
|---|---|---|---|
| **VERIFICATION SUCCESSFUL** | **24** | see list below | Correct driver behavior — all assertions hold, invariants preserved |
| **VERIFICATION FAILED** | **15** | see list below | Intentional bugs — driver violates spec/contract |

**SUCCESSFUL (24):** `cxl_aer_01`, `cxl_driver_aer_fatal_01`, `cxl_error_01`,
`cxl_hdm_01`, `cxl_driver_hdm_align_01`, `cxl_driver_probe_01`,
`cxl_mailbox_state_01`, `cxl_security_01`, `cxl_mem_attach_01`,
`cxl_partition_01`, `cxl_mmio_01`, `cxl_mmio_readback_01`, `cxl_pci_enum_01`,
`cxl_port_enum_01`, `cxl_irq_01`, `cxl_driver_irq_01`, `cxl_device_init_01`,
`cxl_concurrent_01`, `cxl_memdev_01`, `cxl_region_01`, `cxl_mbox_ioctl_01`,
`cxl_port_dport_01`, `cxl_mbox_race_01`, `cxl_dma_coherent_01`

**FAILED (15):** `cxl_irq_02` (double-free), `cxl_hdm_overlap_01` (missing
overlap check), `cxl_driver_hdm_align_fail_01` (missing alignment check),
`cxl_driver_remove_01` (missing IRQ cleanup), `cxl_mailbox_01` (unchecked
return value), `cxl_security_02` (invalid state transition), `cxl_dma_01`
(missing DMA sync), `cxl_pci_enum_02` (NULL pointer dereference),
`cxl_mmio_readback_02` (unwritten register assumed fixed),
`cxl_memdev_02` (unchecked id allocation), `cxl_region_02` (missing overlap
check), `cxl_mbox_ioctl_02` (unchecked payload size), `cxl_port_dport_02`
(use-after-free), `cxl_mbox_race_02` (unsynchronised mailbox),
`cxl_dma_coherent_02` (unsynchronised DMA buffer)

40 + 31 = 71, matching `ctest -L cxl` with `-DENABLE_CXL_REGRESSION=On`.

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

1. **Roadmap statistics** are now 40 passing / 31 bug-detecting across 71 suites. Note the figure that matters more: all 117 modelled functions are now exercised by some test, up from 22 of 105 (`scripts/cxl_model_coverage.py`). Test count is not coverage.

2. **Flags are adequate but sparse.** No test uses:
   - `--z3` / `--bitwuzla` solver selection — all tests run with the default SMT solver
   - `--no-bmc` — not needed (BMC is the default and correct mode for these tests)
   - `--floats` — no floating-point code in any test
   - `--pointer-check` — pointer bugs are tested via `assert` (e.g., pci_enum_02), not the built-in pointer checker. For deeper pointer analysis (buffer overflows, OOB), this could be added.
   - `--data-races-check` — now used by the five concurrency tests. The claim in an earlier revision, that `__ESBMC_atomic_begin/end` was "simpler and sufficient", was mistaken: `cxl_concurrent_01` was single-threaded, so there were no interleavings for either approach to explore.

3. **cxl_driver_irq_01** uses `__VERIFIER_nondet_int() % 3` to explore all 3 IRQ types over a 10-iteration loop — small enough that BMC unwinding won't explode. `cxl_irq_01`, despite the similar name, is fully deterministic and has no loop.

4. **No tests use `--k-induction` or `--unwind N`.** For the small programs here, default BMC with automatic unwinding is appropriate. If tests grow to include unbounded loops, k-induction would become necessary.
