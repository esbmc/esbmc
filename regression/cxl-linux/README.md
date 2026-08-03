# Verifying real Linux CXL driver source

Unlike `regression/cxl/`, which is a synthetic suite against an invented
CXL-like API, the harnesses here compile and verify **actual Linux kernel
source** from `drivers/cxl/`. They need a kernel tree and a configured kernel
build on the machine, so they are deliberately **not registered with ctest**
(`regression/CMakeLists.txt` lists suites explicitly) and do not run in CI.

## Prerequisites

A kernel source tree (developed against Linux 7.1.5) and generated headers.
Configure **out of tree** so the source stays pristine:

```sh
LINUX=/path/to/linux-7.1.5
BUILD=/path/to/linux-build-cxl

make -C "$LINUX" O="$BUILD" ARCH=x86_64 defconfig
"$LINUX"/scripts/config --file "$BUILD/.config" \
    -e CXL_BUS -e CXL_PCI -e CXL_MEM -e CXL_PORT -e CXL_ACPI
make -C "$LINUX" O="$BUILD" ARCH=x86_64 olddefconfig
make -C "$LINUX" O="$BUILD" ARCH=x86_64 prepare
```

`make prepare` fails at `objtool` unless `libelf` headers are installed. That
failure is harmless here: every generated header ESBMC needs
(`include/generated/autoconf.h`, `asm-offsets.h`, the `arch/x86/include/generated`
wrappers) is produced before objtool is built.

Point `run_esbmc.sh` at both directories by editing `L` and `B`, then:

```sh
./run_esbmc.sh harness_cdat_checksum.c --unwind 12 --no-unwinding-assertions
```

## Why the flags are needed

| Flag | Reason |
|---|---|
| `--fms-extensions` | `struct filename` (`include/linux/fs.h`) uses an unnamed field of a named struct type |
| `-D__KERNEL__`, `-DKBUILD_MODNAME` | expected by kernel headers and `EXPORT_SYMBOL()` |
| generated-header includes | the kernel does not build without `autoconf.h` and friends |

The three `-include` headers the kernel Makefile forces (`compiler-version.h`,
`kconfig.h`, `compiler_types.h`) are pulled in at the top of each harness
instead, because ESBMC has no `-include` (its equivalent is `--include-file`).

## Harnesses

| Harness | Target | Expected |
|---|---|---|
| `harness_core_pci.c` | whole file, conversion only | GOTO program, no verification |
| `harness_cdat_checksum.c` | `cdat_checksum()` | SUCCESSFUL |
| `harness_cdat_checksum_fail.c` | `cdat_checksum()` with the buffer bound relaxed | FAILED — array bounds violated at `pci.c:554` |
| `harness_latency_nocontract.c` | `cxl_pci_get_latency()` | FAILED — division by zero at `pci.c:667` |
| `harness_latency_contract.c` | same, callee contract modelled | SUCCESSFUL |
| `harness_dvsec_rr_decode.c` | `cxl_dvsec_rr_decode()` | SUCCESSFUL |
| `harness_dvsec_rr_decode_fail.c` | same, asserting `ranges <= 1` | FAILED — 2 ranges are reachable |
| `harness_hdm_decode_init.c --unwind 3` `-DRANGES_BOUNDED` | `cxl_hdm_decode_init()` | SUCCESSFUL |
| `harness_hdm_decode_init.c --unwind 5` | same, caller contract dropped | FAILED — out of bounds at `range.h:20` |

Run them with `./run_all.sh`, which checks every row of this table and exits
non-zero on any mismatch. A prose table is not a test.

The `cdat_checksum` pair is the two-tier pattern: the positive harness proves
the loop stays in bounds for `size <= sizeof(buf)`, and the negative one relaxes
exactly that bound so the checker is shown to be live rather than vacuous.

Run the `cdat_checksum` pair **without** `--no-unwinding-assertions` as well.
Both still report SUCCESSFUL, which is what makes the bound sound rather than
an artefact of truncated unwinding.

The HDM pair needs `--unwind` for a less obvious reason. Both harnesses cap
`info.ranges` with `__ESBMC_assume()`, but an assumption constrains the
*value*, not the *unwinding*: symbolic execution still walks
`for (i = 0; i < info->ranges; i++)` (`pci.c:428`) syntactically, and with no
bound it does so forever — an unbounded run reached iteration 4562 and
exhausted 12 GB before failing. Each bound below is one past the harness's
assumed maximum, and unwinding assertions are left **on**, so ESBMC proves the
loop cannot exceed it. With the bound in place both harnesses finish in under
a second.

## The two DVSEC functions, and the contract between them

`cxl_dvsec_rr_decode()` fills `info->dvsec_range[]`, which holds 2 entries. The
store is kept in bounds solely by the `hdm_count > 2` rejection, so that guard
is load-bearing; the harness proves `info.ranges <= 2` over nondeterministic PCI
config space. The `_fail` variant asserts `ranges <= 1` instead and fails, which
shows both ranges are genuinely reachable.

`cxl_hdm_decode_init()` then walks `info->dvsec_range[i]` for `i < info->ranges`
and never re-validates that bound. It is safe only because its sole caller
(`core/hdm.c:1273`) passes the `info` that `cxl_dvsec_rr_decode()` populated at
`hdm.c:1262`. Drop that assumption and ESBMC reports the out-of-bounds read.
So the two functions are joined by a precondition that appears nowhere in the
code — the same lesson as the latency pair, but inter-procedural.

## Stub surface

Verifying against undefined kernel functions makes every stub an assumption:

- `pci_read_config_word/dword()` — hardware state: unconstrained value, may fail.
- `to_cxl_port()`, `to_cxl_decoder()` — model the success path only, returning a
  well-formed object. The real `to_cxl_port()` can return NULL on a device-type
  mismatch, and the caller in `cxl_hdm_decode_init()` would then dereference it;
  that path is outside these harnesses, which fix a well-typed topology.
- `device_find_child()` — must invoke `match()` rather than return a nondet
  pointer. Stubbing it out would leave `&info->dvsec_range[i]` as mere address
  arithmetic, and the out-of-bounds read would never be exercised.
- `harness_decoder.flags` must carry `CXL_DECODER_F_RAM`, otherwise the match
  callback returns before `range_contains()` and the negative harness passes
  vacuously.

The last two are worth dwelling on: both were real defects in an earlier version
of this harness that made the negative case pass for the wrong reason. A negative
harness that does not fail is not evidence of correctness — it is evidence the
harness is wrong.

## The latency pair is not a kernel bug

`cxl_pci_get_latency()` guards `bw < 0` and then divides by `bw / BITS_PER_BYTE`,
so an unconstrained callee lets `bw` land in `0..7` and the division traps. That
is a false positive: `pcie_dev_speed_mbps()` (`drivers/pci/pci.h`) returns
`-EINVAL` or one of `{2500, 5000, 8000, 16000, 32000, 64000}`, and
`pcie_link_speed_mbps()` otherwise forwards a negative error, so a non-negative
result is always `>= 2500`.

The invariant is real but unstated in the code, which is the point of keeping
both harnesses: verifying real driver code against undefined kernel functions
requires modelling each callee's contract, and every such assumption is a
soundness obligation. `harness_latency_contract.c` shows the assumption written
down explicitly.
