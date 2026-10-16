# Terminal Trace Replay Demo

The replay interface is built into ESBMC using FTXUI v6.1.9. It displays source
locations and recorded current scalar values from a single-threaded C trace.
It does not execute the program again.

## Build and Run

Using an existing build configured with Z3 and the project dependencies:

```sh
cmake -S . -B build -DENABLE_TRACE_REPLAY=ON -DDOWNLOAD_DEPENDENCIES=ON
cmake --build build --target esbmc -j2
build/src/esbmc/esbmc docs/examples/trace-replay/demo.c --z3 --trace-replay
```

With dependency downloads disabled, install FTXUI 6.1.9 or a compatible version
and provide its CMake package path. `ENABLE_TRACE_REPLAY` defaults to off.

## Controls

| Key | Action |
| --- | --- |
| `n` or Right | Next event |
| `p` or Left | Previous event |
| `e` | Jump to the unsatisfied property |
| `q` | Close this trace and resume verification/reporting |

The variable panel shows recorded values after the selected event. The demo
starts with `x = 1`, adds `y = 0`, updates `x = 2`, then reaches the failed
`x == 3` assertion. Going backward restores the earlier event's values.
Quitting preserves `VERIFICATION FAILED` and exit code 1.

Replay enables `--no-slice` internally to retain assignments. To inspect the
same records without the interface:

```sh
build/src/esbmc/esbmc docs/examples/trace-replay/demo.c --z3 --no-slice
```

## Verification Strategies

The same viewer consumes the trace produced by ordinary BMC, serial incremental
BMC, or serial k-induction. The header identifies the strategy, phase, and bound
when available. Use the loop example to find a counterexample after increasing
the bound:

```sh
build/src/esbmc/esbmc docs/examples/trace-replay/loop.c --z3 \
  --incremental-bmc --max-k-step 6 --trace-replay

build/src/esbmc/esbmc docs/examples/trace-replay/loop.c --z3 \
  --k-induction --max-k-step 6 --trace-replay
```

Base-case counterexamples open normally. To also inspect available forward
condition or inductive-step models, add `--show-cex`, following ESBMC's existing
trace-generation policy. Such traces are labeled as proof-phase models, not
confirmed program counterexamples. `q` closes the current trace; verification
can continue and open a later trace. No trace means no replay session.

Redirected input/output uses ordinary text output instead of opening the
interface. Parallel k-induction, multi-property reporting, coverage,
`--result-only`, `--ltl`, `--smt-during-symex`, and `--file-output` remain
unsupported by this demo.

The current scope is scalar values in a simple single-threaded trace. Values
not recorded in the trace remain unknown. Arrays, pointer memory, recursive
call instances, and variable lifetime tracking are not implemented.

## Recorded Validation

The demo was exercised in a real 120-column terminal session with
`n`, `n`, `p`, `e`, `q`. Source highlighting, current values, backward navigation,
failure selection, terminal restoration, and failure exit status were checked.
The development run produced these local build artifacts:

- `build/trace-replay-demo.cast`: terminal recording in asciinema v2 format.
- `build/trace-replay-demo.txt`: decoded screen snapshots.

These artifacts are local outputs, not required build inputs. Broader automated
validation is deferred beyond the minimal demo.

The loop demo was also checked in real terminal sessions with incremental BMC
and k-induction: both found the counterexample at `k = 4`. Forward-condition and
inductive-step models were checked separately with `--unwind 2 --show-cex`.
Their proof-model labels, terminal restoration, and unchanged exit status were
confirmed. Screens are saved locally in `build/trace-replay-strategies.txt`, with
per-session `.cast` recordings under `build/trace-replay-*.cast`.
