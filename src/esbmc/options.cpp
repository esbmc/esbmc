#include <esbmc/esbmc_parseoptions.h>
#include <fstream>
#include <util/cmdline.h>

const struct opt_templ esbmc_options[] = {

  // Options
  { '?', "", switc, "" },
  { 'h', "", switc, "" },
  { 'I', "", string, "" },
  { 0, "help", switc, "" },

  // Printing options
  { 0, "symbol-table-only", switc, "" },
  { 0, "symbol-table-too", switc, "" },
  { 0, "parse-tree-only", switc, "" },
  { 0, "parse-tree-too", switc, "" },
  { 0, "goto-functions-only", switc, "" },
  { 0, "goto-functions-too", switc, "" },
  { 0, "program-only", switc, "" },
  { 0, "program-too", switc, "" },
  { 0, "ssa-symbol-table", switc, "" },
  { 0, "ssa-guards", switc, "" },
  { 0, "ssa-no-sliced", switc, "" },
  { 0, "ssa-full-names", switc, "" },
  { 0, "ssa-no-location", switc, "" },
  { 0, "smt-formula-only", switc, "" },
  { 0, "smt-formula-too", switc, "" },
  { 0, "show-smt-model", switc, "" },

  // Trace
  { 0, "quiet", switc, "" },
  { 0, "symex-trace", switc, "" },
  { 0, "ssa-trace", switc, "" },
  { 0, "symex-ssa-trace", switc, "" },
  { 0, "show-goto-value-sets", switc, "" },
  { 0, "show-symex-value-sets", switc, "" },

  // Frontend
  { 'I', "", string, "" },
  { 'D', "", string, "" },
  { 0, "preprocess", switc, "" },
  { 0, "no-inlining", switc, "" },
  { 0, "full-inlining", switc, "" },
  { 0, "all-claims", switc, "" },
  { 0, "show-loops", switc, "" },
  { 0, "show-claims", switc, "" },
  { 0, "show-vcc", switc, "" },
  { 0, "document-subgoals", switc, "" },
  { 0, "no-arch", switc, "" },
  { 0, "no-library", switc, "" },
  { 0, "binary", string, "" },
  { 0, "little-endian", switc, "" },
  { 0, "big-endian", switc, "" },
  { 0, "16", switc, "" },
  { 0, "32", switc, "" },
  { 0, "64", switc, "" },
  { 0, "unsigned-char", switc, "" },
  { 0, "version", switc, "" },
  { 0, "witness-output", string, "" },
  { 0, "witness-detailed", switc, "" },
  { 0, "old-frontend", switc, "" },
  { 0, "result-only", switc, "" },
  { 0, "i386-linux", switc, "" },
  { 0, "i386-macos", switc, "" },
  { 0, "i386-win32", switc, "" },
  { 0, "ppc-macos", switc, "" },

  // BMC
  { 0, "function", string, "" },
  { 0, "claim", number, "" },
  { 0, "depth", number, "" },
  { 0, "unwind", number, "" },
  { 0, "unwindset", string, "" },
  { 0, "no-unwinding-assertions", switc, "" },
  { 0, "partial-loops", switc, "" },
  { 0, "unroll-loops", switc, "" },
  { 0, "no-slice", switc, "" },
  { 0, "extended-try-analysis", switc, "" },
  { 0, "skip-bmc", switc, "" },

  // IBMC
  { 0, "incremental-bmc", switc, "" },
  { 0, "falsification", switc, "" },

  // Solver
  { 0, "list-solvers", switc, "" },
  { 0, "boolector", switc, "" },
  { 0, "z3", switc, "" },
  { 0, "mathsat", switc, "" },
  { 0, "cvc", switc, "" },
  { 0, "yices", switc, "" },
  { 0, "bv", switc, "" },
  { 0, "ir", switc, "" },
  { 0, "smtlib", switc, "" },
  { 0, "smtlib-solver-prog", string, "" },
  { 0, "output", string, "" },
  { 0, "floatbv", switc, "" },
  { 0, "fixedbv", switc, "" },

  // Incremental SMT
  { 0, "smt-during-symex", switc, "" },
  { 0, "smt-thread-guard", switc, "" },
  { 0, "smt-symex-guard", switc, "" },

  // Property checking
  { 0, "no-assertions", switc, "" },
  { 0, "no-bounds-check", switc, "" },
  { 0, "no-div-by-zero-check", switc, "" },
  { 0, "no-pointer-check", switc, "" },
  { 0, "no-align-check", switc, "" },
  { 0, "nan-check", switc, "" },
  { 0, "memory-leak-check", switc, "" },
  { 0, "overflow-check", switc, "" },
  { 0, "deadlock-check", switc, "" },
  { 0, "data-races-check", switc, "" },
  { 0, "lock-order-check", switc, "" },
  { 0, "atomicity-check", switc, "" },
  { 0, "error-label", string, "" },
  { 0, "force-malloc-success", switc, "" },

  // k-induction
  { 0, "base-case", switc, "" },
  { 0, "forward-condition", switc, "" },
  { 0, "inductive-step", switc, "" },
  { 0, "k-induction", switc, "" },
  { 0, "k-induction-parallel", switc, "" },
  { 0, "constrain-all-states", switc, "" },
  { 0, "k-step", number, "1" },
  { 0, "max-k-step", number, "50" },
  { 0, "unlimited-k-steps", switc, "" },
  { 0, "show-counter-example", switc, "" },

  // Scheduling
  { 0, "schedule", switc, "" },
  { 0, "round-robin", switc, "" },
  { 0, "time-slice", number, "1" },

  // Concurrency checking
  { 0, "context-bound", number, "-1" },
  { 0, "state-hashing", switc, "" },
  { 0, "control-flow-test", switc, "" },
  { 0, "no-por", switc, "" },
  { 0, "all-runs", switc, "" },

  // Miscellaneous
  { 0, "memlimit", string, "" },
  { 0, "memstats", switc, "" },
  { 0, "timeout", string, "" },
  { 0, "enable-core-dump", switc, "" },
  { 0, "no-simplify", switc, "" },

  // DEBUG options

  // Print commit hash for current binary
  { 0, "git-hash", switc, "" },

  // Finish parallel k-induction if a child dies
  { 0, "dont-ignore-dead-child-process", switc, "" },

  // Check if there is two or more assingments to the same SSA instruction
  { 0, "double-assign-check", switc, "" },

  // Use SMT flatteners instead of the ones provided by the solvers
  { 0, "tuple-node-flattener", switc, "" },
  { 0, "tuple-sym-flattener", switc, "" },
  { 0, "array-flattener", switc, "" },

  // Abort if the program contains a recursion
  { 0, "abort-on-recursion", switc, "" },

  // Don't generate bitfields, all members members will have the full size
  { 0, "no-bitfields", switc, "" },

  // The clang frontend can mark unused struct/functions, in the future this
  // will be enabled by default, but for now, it needs more testing
  { 0, "keep-unused", switc, "" },

  // LTL mode?
  { 0, "ltl", switc, "" },

  // Verbosity of message, probably does nothing
  { 0, "verbosity", number, "" },

  // --break-at $insnnum will cause ESBMC to execute a trap
  // instruction when it executes the designated GOTO instruction number.
  // Now breaks into python, although you can
  // break further into gdb by executing esbmc.trap().
  { 0, "break-at", string, "" },

  // I added some intrinsics along the line of "__ESBMC_switch_to_thread"
  // that immediately transitioned to a particular thread and didn't allow
  // any other exploration from that point. Useful for constructing an
  // explicit multithreading path
  { 0, "direct-interleavings", switc, "" },

  // Used to print out the instructions that had been identified as touching
  // global variables, thus instructions that cause interleavings. Probably
  // isn't sound any more seeing how we don't do the static pointer analysis
  // any more.
  { 0, "show-ileave-points", switc, "" },

  // I think this dumps the current stack of all threads on an ileave point.
  // Useful for working out the state of _all_ the threads and how they
  // evolve, also see next flag,
  { 0, "print-stack-traces", switc, "" },

  // At every ileave point ESBMC stops and asks the user what thread to
  // transition to. Useful again for trying to replicate a particular context
  // switch order, or quickly explore what's reachable.
  { 0, "interactive-ileaves", switc, "" },

  { 0, "", switc, "" }
};
