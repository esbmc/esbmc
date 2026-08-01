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

The `cdat_checksum` pair is the two-tier pattern: the positive harness proves
the loop stays in bounds for `size <= sizeof(buf)`, and the negative one relaxes
exactly that bound so the checker is shown to be live rather than vacuous.

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
