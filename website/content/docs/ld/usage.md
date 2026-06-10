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

When a property is violated, the counterexample names it via the property's
`description` field:

```
Violated property:
  file unsafe.ld ...
  Coils A and B must never be simultaneously energised
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
