# Frozen Foundry environment (stage-4 acceptance)

Stage 4's only criterion that can veto the whole thing is: *a generated test, run
against the unmodified contract, passes.* That cannot be evaluated without an
environment that compiles what the generator emits, so this directory exists
before the generator does.

## Verified, not assumed

Installing forge-std is not the claim. The claim is that something the generator
actually emitted compiles and runs here. Measured 2026-07-29:

```
$ ./smoke.sh foundry_covgen_scalar_fail
Ran 2 tests for test/Cov.cov.t.sol:CovCovTest
[PASS] test_cov_0() (gas: 28354)
[PASS] test_cov_1() (gas: 8452)
2 passed; 0 failed
```

That case is a real `*.cov.t.sol` from the regression tree, using the shape the
generator emits: `pragma solidity >=0.8.0`, `import {Test} from
"forge-std/Test.sol"`, and a RELATIVE `import ... from "./contract.sol"` — which
is why `smoke.sh` copies the contract alongside the test rather than into `src/`.

## Pinned versions

| Component | Pin | Verified value |
|---|---|---|
| forge | (whatever is installed; recorded, not enforced) | `1.7.1`, commit `4072e48705af9d93e3c0f6e29e93b5e9a40caed8`, built 2026-05-08 |
| solc | `foundry.toml: solc_version` | `0.8.34+commit.80d5c536` |
| forge-std | tag **and** commit, checked by `setup.sh` | `v1.16.2` = `bf647bd6046f2f7da30d0c2bf435e5c76a780c1b` |
| optimizer | `foundry.toml` | off |
| evm_version | `foundry.toml` | `cancun` |
| fuzz runs / seed | `foundry.toml` | `256` / `0x1` |
| max_test_rejects | `foundry.toml` | `65536` |

`forge-config.frozen.txt` holds the FULL resolved `forge config` output as of the
run above, so a later divergence is visible instead of silent. The optimizer is
off deliberately: optimiser passes can merge or delete branches, which is the one
thing that would make a path-coverage claim about the source untrue of the
compiled artifact.

`setup.sh` verifies the commit it actually got, not just the tag. A tag can be
moved; a commit cannot. On mismatch it refuses to proceed, because numbers
measured under a different forge-std are not comparable to the ones in the paper.

## `lib/` is not vendored, and why

`lib/` is gitignored. Vendoring would add ~1000 files of forge-std to this
repository, and a submodule would change `.gitmodules` for everyone. The pin
lives in `setup.sh` as an exact tag plus an exact commit, which is reproducible
without either cost. If the project later wants a self-contained artifact, a
submodule is the upgrade path.

## Environment self-test

`test/env/BoundVsAssume.t.sol` measures the pair of facts the method's rendering
rule rests on rather than citing them: `bound` is a mapping that never rejects,
`vm.assume` is a filter that discards draws. That is *why* a box is the only
coordinate shape that executes at full yield, and it is a property of this
forge/forge-std rather than a theorem.

Measured here: both fuzz cases complete `runs: 256` — the configured count, with
nothing discarded. The degenerate box `LO == HI` also runs at full yield, which
matters because a single-point box is what the method falls back to when
generalisation fails. And an EMPTY interval reverts inside `_bound`, which is the
concrete reason criterion 5.0 demands the emptiness check happen BEFORE
rendering: without it the failure surfaces as an unexplained revert inside a
helper instead of as a refused test.

`forge test` failing on that file means the environment no longer supports a
claim the paper makes — which is exactly when we want to hear about it.

## Usage

```
./setup.sh                              # fetch + verify the forge-std pin
./smoke.sh [regression_case_name]       # run a real generated test
forge test --match-path 'test/env/*'    # environment self-test
```

Generated tests are produced by a run, not checked in for every case, so
`smoke.sh` reports plainly when a case has no `*.cov.t.sol` rather than passing
vacuously on an empty set.
