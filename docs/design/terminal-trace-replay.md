# Terminal Trace Replay: Implementation Plan

The minimal demo is implemented. See
[the demo instructions](../examples/trace-replay/README.md) for build and usage.

## Goal

Add a C++ terminal interface to ESBMC that replays a counterexample over the
original C source, shows recorded variable changes, and highlights the failure.
Use [FTXUI](https://arthursonzogni.github.io/FTXUI/ftxui.html) for the interface.

The initial scope is one single-threaded C counterexample with scalar variables.
Replay follows recorded trace events; it does not execute the C program again.
Arrays, pointer memory, multiple traces, and dependency graphs are later work.

## Implementation Steps

1. **Inspect the existing trace.** Reuse `goto_tracet` and `goto_trace_stept` in
   `src/goto-symex/trace/goto_trace.h`. Read source locations through `pc`, concrete
   assignment values through `value`, and failure details through assertion
   steps. Check `build_goto_trace.cpp` and `goto_trace.cpp` for existing helpers.

2. **Add an optional build dependency.** Use `ENABLE_TRACE_REPLAY`, default
   off, and link FTXUI v6.1.9 when enabled. Place replay code under
   `src/esbmc/trace_replay/` and wire it into `src/esbmc/CMakeLists.txt`. Keep build
   directories under `build/`.

3. **Adapt the trace.** Build an indexed event list containing source position,
   event type, variable identity, recorded value, and failure description. Reuse
   existing expression formatting. Keep distinct events on the same source line;
   use symbol identity rather than display names as variable keys.

4. **Reconstruct variable state.** Show only the known variable values after the
   selected event. For the demo, rebuild state from the start when navigating
   backward. Unrecorded values remain unknown.
   A trace may omit operations after simplification or slicing; do not invent
   intermediate values or dependencies.

5. **Build the terminal view.** Load the original source, highlight the current
   line, and show variables in a second panel. Add `n` for next, `p` for previous,
   `e` for failure, and `q` for quit. Display event position and failure text.
   Restore terminal state on exit; show event information if source is missing.

6. **Connect the CLI.** Add the `--trace-replay` option in
   `src/esbmc/options.cpp` and pass it through command-line processing. The
   ordinary `bmct::error_trace()` path in `src/esbmc/bmc.cpp` is the initial
   integration point. Start one session on the main reporting thread after the
   counterexample is built, keeping its data alive throughout replay. Fall back
   to normal output outside an interactive terminal. Quitting replay must
   preserve the verification verdict and exit code. Enable `no-slice` for replay
   to retain assignments; other simplifications can still omit trace events.
   Reuse this path for serial incremental BMC and k-induction, showing the phase
   and bound as metadata. Proof-phase traces requested with `--show-cex` must be
   labeled as models rather than confirmed program counterexamples. Closing a
   trace returns control to the verification strategy.

## Minimal Demo Validation

For this phase, validate only one real C program and its ESBMC-generated trace:

```c
#include <assert.h>

int main(void)
{
  int x = 1;
  int y = x - 1;
  x = y + 2;
  assert(x == 3);
  return 0;
}
```

Enable `ENABLE_TRACE_REPLAY` in the existing build configuration and run:

```sh
build/src/esbmc/esbmc docs/examples/trace-replay/demo.c --z3 --trace-replay
```

Confirm manually that:

- Source highlighting follows the recorded events.
- The variable panel shows the recorded values at each selected event, including
  `x = 1`, `y = 0`, then `x = 2`; stepping backward shows the values at the earlier
  event. Compare with the normal counterexample using `--no-slice` first: if
  an assignment was omitted, the viewer must not fabricate it.
- `e` selects the failed `x == 3` assertion and shows its failure information.
- `q` restores the terminal and retains `VERIFICATION FAILED` and the normal
  failure exit code.

Deliver the source file, build/run commands, and a short terminal recording or
screenshots. Broader automated validation is deferred beyond this demo phase.
