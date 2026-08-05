# Bounded model checking of Linux CXL driver code: methodology and findings

**Status: internal technical report.** This is the write-up called for by
Phase 6.3 of `docs/cxl-driver-verification-roadmap.md`. It has not been
submitted anywhere. §7 sets out what would have to be true first.

---

## 1. What was built

Two artefacts with different claims attached.

**A synthetic suite** — `regression/cxl/`, 84 tests against an operational
model of a CXL-like driver API (`src/c2goto/library/cxl_driver.c`, 127
functions). The API is *invented*: Linux has no `include/linux/cxl.h`, and the
real declarations in `drivers/cxl/cxl.h` use different names. What these tests
establish is that a bug class is expressible and that ESBMC detects it.

**Real-source harnesses** — `regression/cxl-linux/`, 8 harnesses that compile
`drivers/cxl/core/pci.c` from a pinned Linux 7.1.5 tree and verify four
functions: `cdat_checksum()`, `cxl_pci_get_latency()`, `cxl_dvsec_rr_decode()`
and `cxl_hdm_decode_init()`. These say something about the driver.

The gap between "84" and "4" is the honest summary of the work's reach.

## 2. Methodology

**Operational models over stubs.** Hardware interaction returns
nondeterministic values constrained by `__ESBMC_assume()` to the ranges the
specification permits. MMIO writes are stored so subsequent reads return them,
which models read-back of writable registers; DMA-coherent memory lives in a
separate address space so stale-read bugs remain detectable.

**Constraints transcribed, not invented.** Every model function added from
Phase 7 onward is a transcription of a specific function in Linux 7.1.5, cited
in a comment. `cxl_pmem_security_flags()` reproduces the derivation in
`cxl_pmem_get_security_state()` line for line; `eiw_to_ways()` reproduces the
CXL ECN interleave encoding. Where the plan's description disagreed with the
source, the source won and the test was retargeted — six of eleven rows in the
final slice.

**Paired tests.** Each property gets a passing test and a failing partner that
relaxes exactly the guard under test. The pair is its own patch-and-reverify:
the failing variant is the passing one with the modelled bug reintroduced, so
"fails for the right reason" is structural rather than asserted.

**Coverage as the primary metric.** Test count measures output. The tracked
number is how many modelled functions any test actually calls
(`scripts/cxl_model_coverage.py`), specifically excluding functions a test
shadows with its own definition.

**Executable verdicts.** Real-source harness results are encoded in
`run_all.sh` with resource caps, not in a prose table.

## 3. Findings

### 3.1 Defects in the verification infrastructure itself

Raising model coverage from 21% to 100% found **16 defects in the operational
model**. Three meant a function could not have worked for any caller:

| Defect | Consequence |
|---|---|
| `pci_get_device()` divided by zero | The device table had no population API; the counter was never incremented anywhere, so it was permanently 0 |
| AER layer keyed on table index | Callers pass their own `struct pci_dev`, never in that table, so the lookup could never match — all four AER functions were dead |
| `readsl()` defined, never declared | Any caller got an implicit declaration returning `int` |

The rest: a leaked array, a mailbox that overran short reply buffers, a
control register written 64-bit and read 32-bit, four sites where
`nondet_int() % N` produced negative values, an arithmetic overflow, and a
relational pointer comparison across objects (UB) on the path meant to reject
bad pointers.

**None were found by adding tests. All were found by reaching code.** This is
the central methodological result: in a verification harness, unexecuted code
is not merely untested, it is routinely non-functional, and test count cannot
detect that.

### 3.2 A defect in ESBMC

`overflow_arith()` derived signedness from either operand of a shift. Since the
count is typically a signed `int` literal, `(u64)hi << 32` took the signed path
and reported overflow for every result ≥ 2^63 — all representable in `u64`.
This rejects `((u64)hi << 32) | lo`, the standard idiom for assembling a 64-bit
register value, which is pervasive in driver code. Fixed with regression tests
(PR #6684).

A second ESBMC defect surfaced as a consequence: `__builtin_object_size()` on a
VLA let `array_type2t::dyn_sized_array_excp` escape and aborted the run, where
GCC answers "unknown". The function already computed that fallback for
unidentifiable objects; the throwing call simply was not routed into it.

### 3.3 Defects in the recorded results

Making the harness verdict table executable found that the invocation the
README documented could not work (the script resolved the harness path after
`chdir`), and that two verdicts had been recorded without the `--unwind` flags
that produce them — one of which, run as documented, exhausted 12 GB.

Preparing the tree for CI found that the harness results had never been
reproducible outside a single build directory: `CONFIG_CC_HAS_COUNTED_BY` is a
*compiler-capability probe*, so whether kernel structs carry
`__attribute__((__counted_by__(m)))` depends on whichever compiler ran
`defconfig`. ESBMC cannot convert the resulting `CountAttributedType`, and a
freshly prepared tree on the same machine failed all ten checks.

### 3.4 Recurring driver bug shapes

Across 35 bug-detecting tests, essentially every modelled defect is one of two
shapes:

- **A dropped return value.** Requesting a device transition is not the same as
  it having happened. Fifteen tests model this: unlock, device init, mailbox
  send, security set, region alloc, AER query.
- **A confused encoding.** A register field is read as the value it encodes.
  CXL's DVSEC `HDM_COUNT` is two bits and can report 3 while its array holds 2;
  the CFMWS interleave-ways field is an encoding where 8 means *three*.

Both survive testing because both are correct on the common path.

## 4. What the numbers mean

| Metric | Value |
|---|---|
| Synthetic tests | 84 (49 passing / 35 bug-detecting) |
| Model functions exercised | 127 / 127 (100%) |
| Real driver functions verified | **4** |
| Suite runtime | 81 s |

The first three are infrastructure health. **Only the fourth is a claim about
Linux**, and it did not change while the others moved substantially.

## 5. Cost

The dominant cost in real-source verification is not solving — it is the stub
surface. Each undefined kernel function is an assumption, and getting one wrong
produces a harness that passes for the wrong reason. Two such defects occurred:
a `device_find_child()` stub that returned a nondet pointer instead of invoking
the match callback (making an out-of-bounds read unreachable), and a decoder
flag whose absence short-circuited the match before the property was reached.

Solving time was rarely the constraint; when it was, the cause was almost
always modelling shape rather than problem size — a symbolically sized
allocation, or per-device state in a side table rather than the device struct.

## 6. Threats to validity

- **The synthetic API is invented.** 100% coverage of it is 100% of something
  that does not exist. Phase 7 made every *constraint* a transcription from
  7.1.5, but the tests still compile no kernel code.
- **Four functions is a small sample**, all in one file (`core/pci.c`), chosen
  partly because they were tractable.
- **Every stub is an unverified assumption.** They are enumerated in
  `regression/cxl-linux/README.md`; they are not discharged.
- **`__counted_by` is dropped.** This removes a bounds *annotation*, not a
  bounds *check* — it feeds only UBSAN/FORTIFY, both disabled — but it is a
  modification of the source under verification.
- **Bounded, not complete.** Several harnesses need explicit `--unwind`.
  Unwinding assertions are left on so bounds are proved rather than assumed,
  but the results remain bounded.
- **No comparison against other tools**, and no ground-truth bug set: no
  historical CXL CVE or fixed kernel bug was reproduced. The defects found were
  in the verification infrastructure, not in Linux.

## 7. What publication would require

Honestly: more than exists here.

1. **More real driver functions**, across several files and preferably several
   driver families — the template in `docs/pcie-driver-verification-template.md`
   is the vehicle.
2. **A ground-truth evaluation.** Reproduce known, fixed CXL kernel bugs from
   git history and show the harnesses catch them. Without this there is no
   evidence the approach finds *driver* bugs, only that it finds bugs in
   verification models.
3. **Discharged or bounded assumptions**, rather than an enumerated stub list.
4. **A baseline.** Syzkaller, Coccinelle, or Smatch on the same functions.

Items 1 and 2 are the substantive gap. The methodological finding in §3.1 —
that unexecuted verification code is routinely non-functional, and that
coverage rather than test count detects it — is defensible now and is the part
most likely to generalise.

## 8. Reproducing

```sh
cmake -DENABLE_CXL_REGRESSION=On -Bbuild -S . && ctest -L cxl     # synthetic
./scripts/cxl_prepare_kernel.sh ~/linux-7.1.5 ~/linux-build-cxl   # real source
cd regression/cxl-linux && ./run_all.sh
python3 scripts/cxl_model_coverage.py                             # coverage
```

CI runs the harnesses weekly via `.github/workflows/cxl-linux.yml`.
