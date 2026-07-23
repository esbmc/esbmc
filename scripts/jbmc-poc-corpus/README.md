# JBMC PoC corpus

Ten small Java programs in five tiers, used to measure how far ESBMC gets on
JBMC-produced GOTO binaries. See `docs/jbmc-goto-binary-poc-plan.md`.

| Tier | Programs | Exercises |
|---|---|---|
| T1 | `T1Arith`, `T1ArithFail` | integer/boolean arithmetic, static calls |
| T2 | `T2Array`, `T2ArrayFail` | array allocation, indexing, bounds |
| T3 | `T3Object`, `T3ObjectFail` | object allocation, instance fields, null |
| T4 | `T4Virtual`, `T4VirtualFail` | inheritance, virtual dispatch, `instanceof` |
| T5 | `T5Exception`, `T5ExceptionFail` | `try`/`catch`, handler type matching |

Each tier ships a clean program and a `Fail` variant carrying one violated
property, so verdict matching is falsifiable in both directions. The plan asks
for both properties in a single program; separate variants are used instead
because a mixed program always yields `FAILED` and cannot distinguish "failed
for the intended reason" from "failed for another one".

Every `Fail` program was checked to fail for its *intended* property:
assertion (T1, T4, T5), array bounds (T2), null dereference (T3).

## Running

```sh
PATH=/path/to/jdk/bin:$PATH ESBMC=/path/to/esbmc ./run-corpus.sh
```

Prints one row per program with JBMC's verdict, ESBMC's, and whether they
agree. JBMC's verdicts are the reference; a disagreement is an ESBMC finding,
not a corpus defect.

## Reference verdicts

Measured with jbmc 6.8.0, JDK 26.0.1, `--unwind 6`, lazy class loading against
`core-models.jar`. JBMC matches the intended verdict on **10/10**.

ESBMC (8.4.0, `24cdc4da31`) currently reaches **0/10**: every program is
blocked by the same construct, `@class_identifier` typed `string`
(plan §2.3.1). One construct, ten programs — that is the Phase 1 work-list.

## Why lazy class loading

Plan §2.2 calls `--no-lazy-methods` load-bearing, because lazy loading is
jbmc's default and omits reachable method bodies. That is right in general and
wrong for this corpus. Measured on a trivial `main`:

| Configuration | goto lines | `cprover_string` refs |
|---|---|---|
| `--no-lazy-methods` + models jar | 915018 | 1876 |
| lazy (default) + models jar | 1182 | **0** |

`--no-lazy-methods` loads the whole jar, including `java.lang.String` and the
CProver string primitives that plan §5 names as a hard stop — so it drags in
precisely the machinery a string-free corpus exists to avoid. These programs
call nothing outside themselves, so lazy loading resolves everything they
reach; completeness comes from the programs being self-contained rather than
from the flag. Code that does call library methods should keep the flag on and
accept the contamination.

## Constraints on new corpus programs

- **No strings.** `main(String[] args)` is unavoidable, but its signature alone
  pulls in no string solver under lazy loading (measured: 0 references).
- **No division by zero, even when caught.** JBMC checks
  `integer-divide-by-zero` as a property in its own right, so a deliberately
  caught `ArithmeticException` still fails verification. An earlier T5 draft hit
  exactly this and was rewritten to use user-defined exceptions.
- **Keep them self-contained**, so lazy loading stays sufficient.
- **Declare `package jbmcpoc;` and name the negative variant `<Tier>Fail`**, not
  `<Tier>_fail`. `run-corpus.sh` derives the expected verdict from the `Fail`
  suffix and qualifies the class name with the package before handing it to
  jbmc, so both are load-bearing rather than cosmetic.
