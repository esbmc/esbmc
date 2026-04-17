#ifndef GOTO_PROGRAMS_GOTO_LOOP_INVARIANT_H_
#define GOTO_PROGRAMS_GOTO_LOOP_INVARIANT_H_

#include <goto-programs/goto_functions.h>
#include <goto-programs/goto_loops.h>
#include <util/guard.h>
#include <util/context.h>
#include <irep2/irep2_expr.h>
#include <set>
#include <unordered_set>
#include <vector>

// Forward declaration: full definition is in frame_enforcer.h (included in .cpp)
class frame_enforcert;

/// \brief Entry point: process loop invariants for all functions.
/// When use_frame_rule is true, enables the Operational Frame Rule
/// (Snapshot → Havoc → Assume) for enhanced inductive verification.
void goto_loop_invariant(
  goto_functionst &goto_functions,
  contextt &context,
  bool use_frame_rule);

/**
 * Combined loop-invariant + k-induction mode.
 *
 * Inserts a non-deterministic "Branch 1" verification block immediately
 * before each annotated loop.  Branch 1 checks that the user-supplied
 * invariant is actually inductive (base case + inductive step), using
 * specially tagged ASSERT instructions so the verifier can distinguish
 * invariant failures from ordinary program assertion failures.
 *
 * The original loop is left untouched for subsequent transformation by
 * goto_k_induction (Branch 2), which adds ASSUME(invariant) after its
 * nondet-assign step to tighten the inductive-step search space.
 */
void goto_loop_invariant_combined(goto_functionst &goto_functions);

class goto_loop_invariantt : public goto_loopst
{
public:
  goto_loop_invariantt(
    const irep_idt &_function_name,
    goto_functionst &_goto_functions,
    goto_functiont &_goto_function,
    contextt &_context,
    bool _use_frame_rule)
    : goto_loopst(_function_name, _goto_functions, _goto_function),
      context(_context),
      use_frame_rule(_use_frame_rule)
  {
    if (function_loops.size())
      goto_loop_invariant();
  }

protected:
  contextt &context;
  bool use_frame_rule;

  /// Frame enforcer shared between the havoc step (ASSUME) and the inductive
  /// step (ASSERT compliance check) for the currently-processed loop.
  /// Allocated in insert_havoc_and_assume_before_condition, freed in
  /// convert_loop_with_invariant after insert_inductive_step_and_termination.
  frame_enforcert *active_frame_enforcer = nullptr;
  /// Assigns targets for the current loop (mirrors loop_assigns passed to
  /// insert_havoc_and_assume_before_condition, kept for use in the ASSERT step).
  std::vector<expr2tc> active_loop_assigns;
  /// Maximum number of instructions to search backwards from the loop head
  /// when locating the LOOP_INVARIANT instruction.  A typical for-loop init
  /// (DECL + ASSIGN for the counter) contributes 2 steps, leaving ample room
  /// for up to ~4 extra declarations before the invariant.  Both
  /// extract_loop_invariants and extract_and_remove_side_effects use this
  /// same limit so their searches are consistent.
  static constexpr size_t kMaxInvariantSearchBack = 10;

  void goto_loop_invariant();

  void convert_loop_with_invariant(loopst &loop);

  // Extract loop invariants from LOOP_INVARIANT instructions
  std::vector<expr2tc> extract_loop_invariants(const loopst &loop);

  // Extract loop assigns targets from LOOP_INVARIANT instructions
  std::vector<expr2tc> extract_loop_assigns(const loopst &loop);

  /** Collect all symbol names reachable from an expression tree. */
  static void collect_symbols(const expr2tc &expr, std::set<irep_idt> &symbols);

  /**
   * Collect DECL/FUNCTION_CALL instructions immediately before LOOP_INVARIANT
   * that define temporaries used in the invariant.  Remove them from the GOTO
   * program and place them in @p side_effects_out for re-insertion before each
   * ASSERT/ASSUME so they are re-evaluated with the havoc'd variables.
   */
  void extract_and_remove_side_effects(
    goto_programt::targett loop_head,
    const loopst &loop,
    const std::vector<expr2tc> &invariants,
    goto_programt &side_effects_out);

  // Insert ASSERT invariant before loop (prepends side_effects if non-empty)
  void insert_assert_before_loop(
    goto_programt::targett &loop_head,
    const std::vector<expr2tc> &invariants,
    const goto_programt &side_effects);

  // Insert HAVOC and ASSUME before loop condition (inserts side_effects
  // between the HAVOC block and the ASSUME).
  // side_effects is non-const: if frame rule is active, old_snapshot assigns
  // are patched in-place so insert_inductive_step_and_termination (called
  // after this) sees the patched version too.
  void insert_havoc_and_assume_before_condition(
    goto_programt::targett &loop_head,
    const loopst &loop,
    const std::vector<expr2tc> &invariants,
    const std::vector<expr2tc> &loop_assigns,
    goto_programt &side_effects);

  // Insert inductive step verification and loop termination
  // (prepends side_effects before the ASSERT)
  void insert_inductive_step_and_termination(
    const loopst &loop,
    const std::vector<expr2tc> &invariants,
    const goto_programt &side_effects);
};

// ---------------------------------------------------------------------------
// Combined mode: Branch 1 invariant verification pass
// ---------------------------------------------------------------------------

/**
 * Handles the Branch 1 (invariant inductivity check) for one function in
 * the combined --loop-invariant + --k-induction mode.
 *
 * For each loop that carries a LOOP_INVARIANT annotation the pass inserts
 * a non-deterministic verification branch *before* the original loop:
 *
 *   IF !nondet_bool() GOTO original_loop_head   [skip Branch 1]
 *   ASSERT(INV)          [base case  – tagged property="invariant-base-case"]
 *   HAVOC(loop_vars)
 *   ASSUME(INV)
 *   ASSUME(loop_entry_cond)
 *   <copy of one loop-body iteration>
 *   ASSERT(INV)          [step case – tagged property="invariant-inductive-step"]
 *   ASSUME(false)        [terminate Branch 1 – never fall through]
 *   original_loop_head:  [unchanged; k-induction will transform this]
 *
 * The branch uses std::list::splice() rather than insert_swap() so that
 * existing backward-GOTO targets (which point to original_loop_head) are
 * not disturbed.
 *
 * Loops whose bodies contain forward GOTOs that jump outside the loop are
 * skipped (too complex to copy safely); they fall back to the ASSUME-only
 * k-induction acceleration provided by goto_k_inductiont.
 */
class goto_loop_invariant_combinedt : public goto_loopst
{
public:
  goto_loop_invariant_combinedt(
    const irep_idt &_function_name,
    goto_functionst &_goto_functions,
    goto_functiont &_goto_function)
    : goto_loopst(_function_name, _goto_functions, _goto_function)
  {
    if (!function_loops.empty())
      process_loops_combined();
  }

private:
  void process_loops_combined();

  /// Insert the Branch 1 verification block before @p loop.
  void insert_invariant_verification_branch(loopst &loop);

  /**
   * Copy the loop body instructions (from the instruction immediately after
   * @p loop_head up to, but not including, @p loop_exit) into @p out.
   *
   * Returns false and leaves @p out empty when the loop body contains a
   * forward GOTO whose target is outside the loop body – such loops are too
   * complex to inline safely and Branch 1 is skipped for them.
   */
  void copy_loop_body(
    goto_programt::targett loop_head,
    goto_programt::targett loop_exit,
    goto_programt &out) const;
};

#endif /* GOTO_PROGRAMS_GOTO_LOOP_INVARIANT_H_ */
