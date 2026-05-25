#ifndef GOTO_PROGRAMS_GOTO_LOOP_TRANSFORM_H_
#define GOTO_PROGRAMS_GOTO_LOOP_TRANSFORM_H_

#include <goto-programs/goto_functions.h>
#include <goto-programs/goto_loops.h>

/// Abstract base for passes that visit every loop in every function and
/// optionally transform it.
///
/// Concrete subclasses (k-induction, termination) override the virtual
/// hooks. The base owns the iteration scaffold:
///
///   1. For each function in `goto_functions`, ask `visit_function`
///      whether to enter it. Default: skip functions without a body.
///   2. For each loop in the function (via `goto_loopst`), ask
///      `should_transform_loop` whether to process it. Default: skip
///      loops whose modified-var set is empty.
///   3. If yes, call `transform_loop`.
///   4. After all loops in the function have been visited, call
///      `after_function`.
///   5. After all functions, call `goto_functions.update()` and
///      `finalize`.
///
/// This shape mirrors what `goto_k_induction(goto_functionst&)` and
/// `goto_termination(goto_functionst&)` do today. The base centralises
/// the iteration so the subclasses don't reinvent it, and so it's clear
/// where each step of "find loops, decide whether to act, act, clean
/// up" lives.
class goto_loop_transformt
{
public:
  explicit goto_loop_transformt(goto_functionst &_goto_functions)
    : goto_functions(_goto_functions)
  {
  }

  virtual ~goto_loop_transformt() = default;

  /// Run the pass: iterate every function and every loop, invoking
  /// the hooks. Subclasses normally call this once from their own
  /// constructor or a public `run()` method.
  void run();

protected:
  goto_functionst &goto_functions;

  /// Return true to enter this function, false to skip it. The
  /// default skips functions without a body. Subclasses commonly
  /// extend this to skip `__ESBMC_main` or library helpers (via
  /// `body.hide`).
  virtual bool visit_function(
    const irep_idt &function_name,
    const goto_functiont &function) const;

  /// Return true to call `transform_loop` for this loop. The default
  /// skips loops whose modified-loop-var set is empty — matching
  /// `goto_k_inductiont::goto_k_induction`'s long-standing behaviour.
  virtual bool should_transform_loop(const loopst &loop) const;

  /// Transform a single loop. Pure: every concrete subclass overrides.
  /// `goto_function` is the function currently being processed (a
  /// non-const reference because the transformation will mutate its
  /// body).
  virtual void transform_loop(
    const irep_idt &function_name,
    goto_functiont &goto_function,
    loopst &loop) = 0;

  /// Called once after every loop in `function` has been visited.
  /// Subclasses can use this for per-function bookkeeping or to insert
  /// per-function instrumentation. Default: no-op.
  virtual void after_function(
    const irep_idt & /*function_name*/,
    goto_functiont & /*goto_function*/)
  {
  }

  /// Called once after every function has been visited and
  /// `goto_functions.update()` has run. Default: no-op.
  virtual void finalize()
  {
  }
};

#endif /* GOTO_PROGRAMS_GOTO_LOOP_TRANSFORM_H_ */
