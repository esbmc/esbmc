#ifndef GOTO_PROGRAMS_GOTO_TERMINATION_H_
#define GOTO_PROGRAMS_GOTO_TERMINATION_H_

#include <goto-programs/goto_functions.h>
#include <goto-programs/goto_k_induction.h>
#include <goto-programs/goto_loop_transform.h>
#include <util/options.h>

/// Reduce non-termination to a reachability safety property.
///
/// Two transformations:
///
///   1. Apply k-induction's per-loop havoc + assume-entry-cond to
///      every function except __ESBMC_main, so the inductive step
///      runs from an arbitrary loop state.
///   2. Insert per-loop ASSERT(false) markers on every edge that
///      leaves a loop (top-test exit, break, early-return-as-goto,
///      goto-out-of-loop, do-while fall-through). Each marker is
///      flagged inductive_step_instruction so base case and forward
///      condition skip it; only the inductive step sees the claim.
///
/// Reduction: the program does NOT terminate iff every marker is
/// unreachable in the inductive step. UNSAT at the marker proves
/// non-termination; SAT in the base case refutes it.
///
/// Soundness gate: if any loop's modified-var set is empty,
/// k-induction's make_nondet_assign inserts no havoc for it
/// (e.g. loops that only write through dereferenced pointers,
/// `*p = ...`). In that case IS would run the loop concretely and a
/// "marker unreachable within k unwindings" UNSAT is no longer a
/// real non-termination witness. The pass sets options'
/// "disable-inductive-step" so the BMC driver treats IS as
/// inconclusive for the program.
void goto_termination(goto_functionst &goto_functions, optionst &options);

class goto_terminationt : public goto_loop_transformt
{
public:
  goto_terminationt(goto_functionst &_goto_functions, optionst &_options)
    : goto_loop_transformt(_goto_functions),
      kinduction(_goto_functions),
      options(_options)
  {
  }

protected:
  /// Skip __ESBMC_main and library helpers (body.hide).
  bool visit_function(
    const irep_idt &function_name,
    const goto_functiont &function) const override;

  /// Accept every loop. Unlike k-induction we don't skip empty-
  /// modified-var loops here: we still want to know about them so we
  /// can track that the IS will be unreliable.
  bool should_transform_loop(const loopst & /*loop*/) const override
  {
    return true;
  }

  /// Delegate the havoc + entry-cond transformation to k-induction
  /// when the loop has modified vars; track when it doesn't.
  void transform_loop(
    const irep_idt &function_name,
    goto_functiont &goto_function,
    loopst &loop) override;

  /// After all loops in this function have been visited by
  /// transform_loop, do the per-loop marker pass for the function.
  /// Doing it here (rather than during transform_loop) lets us
  /// process all of the function's loops together — they need to be
  /// sorted innermost-first, and iterator math is easier when we
  /// don't intersperse marker inserts with k-induction's transform.
  void after_function(
    const irep_idt &function_name,
    goto_functiont &goto_function) override;

  /// Set the disable-inductive-step option if any loop in any
  /// function would make IS UNSAT not a real non-termination
  /// witness (see transform_loop for the cases).
  void finalize() override;

private:
  goto_k_inductiont kinduction;
  optionst &options;

  /// True iff at least one loop would make IS UNSAT unsound as a
  /// non-termination witness. When true, finalize() sets
  /// "disable-inductive-step" for the whole program.
  bool any_unreliable_is_loop = false;

  void insert_markers_for_function(
    const irep_idt &function_name,
    goto_functiont &goto_function);
};

#endif /* GOTO_PROGRAMS_GOTO_TERMINATION_H_ */
