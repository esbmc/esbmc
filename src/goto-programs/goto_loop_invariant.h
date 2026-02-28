#ifndef GOTO_PROGRAMS_GOTO_LOOP_INVARIANT_H_
#define GOTO_PROGRAMS_GOTO_LOOP_INVARIANT_H_

#include <goto-programs/goto_functions.h>
#include <goto-programs/goto_loops.h>
#include <util/guard.h>
#include <irep2/irep2_expr.h>
#include <set>

void goto_loop_invariant(goto_functionst &goto_functions);

class goto_loop_invariantt : public goto_loopst
{
public:
  goto_loop_invariantt(
    const irep_idt &_function_name,
    goto_functionst &_goto_functions,
    goto_functiont &_goto_function)
    : goto_loopst(_function_name, _goto_functions, _goto_function)
  {
    if (function_loops.size())
      goto_loop_invariant();
  }

protected:
  /// Maximum number of instructions to search backwards from the loop head
  /// when locating the LOOP_INVARIANT instruction or its extracted side
  /// effects.  Both extract_loop_invariants and extract_and_remove_side_effects
  /// use this same limit so their searches are consistent.
  static constexpr size_t kMaxInvariantSearchBack = 10;

  void goto_loop_invariant();

  void convert_loop_with_invariant(loopst &loop);

  // Extract loop invariants from LOOP_INVARIANT instructions
  std::vector<expr2tc> extract_loop_invariants(const loopst &loop);

  /** Collect all symbol names reachable from an expression tree. */
  static void collect_symbols(const expr2tc &expr, std::set<irep_idt> &symbols);

  /**
   * Collect DECL/FUNCTION_CALL instructions immediately before LOOP_INVARIANT
   * that define temporaries used in the invariant. Remove them from the GOTO
   * program and return in original order for re-insertion before each
   * ASSERT/ASSUME.
   */
  void extract_and_remove_side_effects(
    goto_programt::targett loop_head,
    const std::vector<expr2tc> &invariants,
    goto_programt &side_effects_out);

  // Insert ASSERT invariant before loop (prepends side_effects if non-empty)
  void insert_assert_before_loop(
    goto_programt::targett &loop_head,
    const std::vector<expr2tc> &invariants,
    const goto_programt &side_effects);

  // Insert HAVOC and ASSUME before loop condition (inserts side_effects
  // between the HAVOC block and the ASSUME)
  void insert_havoc_and_assume_before_condition(
    goto_programt::targett &loop_head,
    const loopst &loop,
    const std::vector<expr2tc> &invariants,
    const goto_programt &side_effects);

  // Insert inductive step verification and loop termination
  // (prepends side_effects before the ASSERT)
  void insert_inductive_step_and_termination(
    const loopst &loop,
    const std::vector<expr2tc> &invariants,
    const goto_programt &side_effects);
};

#endif /* GOTO_PROGRAMS_GOTO_LOOP_INVARIANT_H_ */
