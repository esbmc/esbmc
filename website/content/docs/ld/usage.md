---
title: Usage
weight: 2
---

## Building with the LD frontend

The Ladder Diagram frontend is disabled by default. Enable it at configure time
with `-DENABLE_LD_FRONTEND=On`, alongside at least one SMT solver:

```sh
cmake -GNinja -Bbuild -S . \
  -DDOWNLOAD_DEPENDENCIES=On \
  -DENABLE_LD_FRONTEND=On \
  -DENABLE_Z3=On \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo

ninja -C build
```

See the [Build Guide](/docs/development/building) for the full dependency list
and solver options.

## Inputs

You provide two files:

- A **Ladder Diagram program** exported as PLCopen XML, given to ESBMC with the
  `.ld` extension (the frontend dispatches on the file extension).
- A **property file** in YAML describing the safety properties to check, passed
  with `--ld-props`. See [Property Format](/docs/ld/property-format).

## Running ESBMC

Prove a property holds on every scan cycle with k-induction:

```sh
esbmc motor_interlock.ld --ld-props props.yaml --k-induction --unlimited-k-steps
```

```
VERIFICATION SUCCESSFUL
```

Search for a violation within a bounded number of scan cycles with BMC:

```sh
esbmc unsafe.ld --ld-props props.yaml --incremental-bmc
```

```
VERIFICATION FAILED
```

Incremental BMC raises the unwind bound automatically, so no explicit
`--unwind` is needed for the infinite scan loop, and unwinding assertions are
disabled during its base case — `VERIFICATION FAILED` therefore always
indicates a genuine property violation, never an artifact of the bound.

When a property is violated, the counterexample names it via the property's
`description` field:

```
Violated property:
  file unsafe.ld ...
  Coils A and B must never be simultaneously energised
```

## User function-block translation

When a program contains user-defined function blocks with Structured Text (ST)
bodies, the frontend translates and inlines them into the scan. Two flags tune
this:

- `--ld-sound-mode` — translate ST bodies in sound Boolean/integer mode:
  constructs the translator cannot lower (function calls, member access) make
  the block body fall back to a **no-op** instead of an over-approximated
  nondeterministic result. This removes over-approximation and yields zero false
  positives, at the cost of not modelling the unsupported construct's effect.

## Scan-overrun watchdog

An optional watchdog models a PLC scan overrun by bounding the iterations of
`WHILE` loops inside user function-block bodies:

- `--ld-scan-watchdog` — instrument those loops with an assertion that fails once
  a loop exceeds the budget. This **changes the verified model**, so enable it
  only when you want to check for scan overruns.
- `--ld-scan-budget=N` — tolerated iterations before the watchdog assertion fails
  (default `8`). Keep `N` at or below the BMC `--unwind` so the assertion stays
  reachable.

```sh
esbmc loop.ld --ld-props props.yaml --ld-scan-watchdog --ld-scan-budget 8 \
  --k-induction --unlimited-k-steps --unwind 11
```

## Reading the result

| ESBMC output | Strategy | Meaning |
|---|---|---|
| `VERIFICATION SUCCESSFUL` | k-induction | The property holds on every scan cycle (proved). |
| `VERIFICATION SUCCESSFUL` | BMC | No violation up to the unwind bound (not a full proof). |
| `VERIFICATION FAILED` | either | A genuine property violation was found; see the counterexample. |

## The `ld-verify` helper

The repository also ships a small `ld-verify` CLI (built whenever
`ENABLE_LD_FRONTEND` is on) that wraps the above: it locates the `esbmc` binary,
accepts a PLCopen `.xml` or `.ld` input, runs the chosen strategy, and prints a
JSON report. It is convenient for tooling and CI; the `esbmc` invocations above
remain the canonical interface.
