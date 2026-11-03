# Regression tests

Rules for adding and running tests under `regression/`. The quickstart ctest
commands and the 10-minute suite cap live in AGENTS.md (Testing).

Regression test format (`test.desc`): line 1 is
`CORE`/`KNOWNBUG`/`FUTURE`/`THOROUGH` (THOROUGH is Linux-only), line 2 is the
source file, line 3 is ESBMC flags, line 4+ are expected output regexes. Every
PR adds a _pair_ of regression tests — see "Regression tests come in pairs"
below.

**Regression tests come in pairs, and both must bite.** A PR that changes
verification behaviour adds **two** regression tests over the same construct:
one pinning `^VERIFICATION SUCCESSFUL$` and one pinning
`^VERIFICATION FAILED$`. Adding only the passing half is the single most
common review rejection on this repo — write the failing counterpart at the
same time, not after a reviewer asks. For a false alarm, the pair is the
reproducer expecting `SUCCESSFUL` and a variant with a real defect expecting
`FAILED`. For a crash, it is the crashing input, now reaching its correct
verdict, and a variant that expects the opposite one.

**Mutation-check every regression test you add.** Revert the source fix,
re-run each new test, and confirm it _changes verdict_; then restore the fix.
A test that passes both before and after pins nothing. This bites most often
when the property never reaches the code you changed — clang already inserted
the cast, the assertion was constant-folded away, the claim was never
generated. Check `--show-claims` output before and after: if it is
byte-identical, the test is not a gate. If no end-to-end test can be made to
bite, say so explicitly in the PR and name what does pin the change (e.g. a
unit test).

**Pin the mode in every test that depends on it.** A `test.desc` with a blank
flags line inherits whatever LLVM ESBMC was built against — `gnu++17` for
ESBMC's bundled clang, `gnu++14` for Apple clang. A test relying on that is not
pinning anything. Give it `--std c++NN` and mutation-check the pin: change it to
an older mode and confirm the test _fails_.

**Per-test budget, and catching slowdowns.** Each test's real budget is
`ESBMC_REGRESS_TIMEOUT` (default 1200s), baked into the test's ctest
environment at configure time. To narrow it for one run without
re-configuring, set `ESBMC_REGRESS_TIMEOUT_MAX`:

```sh
ESBMC_REGRESS_TIMEOUT_MAX=45 ctest -j$(nproc) -L loop-invariants
```

A `CORE`/`THOROUGH` test slower than the cap then fails, naming the cap in
its message. Without it the suite is blind to performance regressions: a
test that went from sub-second to nine minutes still reports `Passed`
(#7628). Use it when a change could affect solve time.

Two exceptions, both of which read as green:

- `KNOWNBUG` and `FUTURE` tests treat a timeout as satisfying the
  expectation (the `FAIL_MODES` branch in `regression/testing_tool.py`), so
  a cap cannot measure them at all. They print `accepted under KNOWNBUG` and
  pass. Grep for that line before reading a capped run as a clean bill of
  health.
- `REQUIRES long_timeout` tests are skipped once the effective budget is
  under 600s, matching how CMake grants the capability in
  `regression/CMakeLists.txt`. A capped run does not measure
  `floats-regression/nn-logistic_5_unsafe`.

**/tmp disk space.** C and C++ runs write nothing to `/tmp`: bundled clang
headers, the C++ operational models and the internal libc are registered with
`file_operations::filesystemt` and served to clang out of `.rodata` via
`esbmc_clang_vfs()` (`src/clang-c-frontend/AST/vfs_adapter.h`). The Python and
Solidity frontends extract to `/tmp`, because they fork `python3`/`solc` and a
separate process cannot read ESBMC's memory. Clean up after large runs of those
suites: `rm -rf /tmp/esbmc*`

`regression/esbmc/bundled_headers_from_vfs` and
`regression/esbmc-cpp/cpp/om_source_from_vfs` pin this: the first asserts clang
is handed `-isystem /esbmc-vfs/libc/headers` and
`-resource-dir /esbmc-vfs/clang`, the second that an OM source location in a
counterexample reads `/esbmc-vfs/cpp/vector`. Reintroducing extraction turns
those paths back into a temp directory and both fail. Note that asserting the
temp directory is _empty_ after a run would not work: `tmp_path`'s destructor
removes what it created, so a run that extracts and cleans up is
indistinguishable from one that never extracted.
