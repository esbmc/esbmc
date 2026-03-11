#ifndef GOTO_PROGRAMS_GOTO_K_INDUCTION_H_
#define GOTO_PROGRAMS_GOTO_K_INDUCTION_H_

#include <goto-programs/goto_functions.h>
#include <goto-programs/goto_loops.h>
#include <util/guard.h>
#include <irep2/irep2_expr.h>
#include <vector>

void goto_k_induction(goto_functionst &goto_functions);

void goto_termination(goto_functionst &goto_functions);

class goto_k_inductiont : public goto_loopst
{
public:
  goto_k_inductiont(
    const irep_idt &_function_name,
    goto_functionst &_goto_functions,
    goto_functiont &_goto_function)
    : goto_loopst(_function_name, _goto_functions, _goto_function)
  {
    if (function_loops.size())
      goto_k_induction();
  }

protected:
  typedef std::unordered_map<unsigned, bool> marked_branchst;
  marked_branchst marked_branch;

  typedef std::unordered_map<unsigned, guardt> guardst;
  guardst guards;

  void goto_k_induction();

  void convert_finite_loop(loopst &loop);

  bool get_entry_cond_rec(
    const goto_programt::targett &loop_head,
    const goto_programt::targett &after_exit,
    guardst &guards);

  void
  make_nondet_assign(goto_programt::targett &loop_head, const loopst &loop);

  void remove_unrelated_loop_cond(guardst &guards, loopst &loop);

  void assume_loop_entry_cond_before_loop(
    goto_programt::targett &loop_head,
    goto_programt::targett &loop_exit,
    const guardst &guard);

  void adjust_loop_head_and_exit(
    goto_programt::targett &loop_head,
    goto_programt::targett &loop_exit);

  void
  assume_cond(const expr2tc &cond, goto_programt &dest, const locationt &loc);

  /**
   * Search backwards from @p loop_head (no step limit) to find the
   * LOOP_INVARIANT instruction for this loop.  The search terminates at the
   * first LOOP_INVARIANT hit, which is always the one for the current loop
   * because __ESBMC_loop_invariant is placed immediately before its loop.
   * Branch 1 instructions (inserted by goto_loop_invariant_combined before
   * loop_head) are transparently skipped.
   */
  std::vector<expr2tc>
  extract_loop_invariants_before_head(goto_programt::targett loop_head) const;

  /**
   * Insert ASSUME(invariant) instructions at position @p pos.  All inserted
   * instructions are marked inductive_step_instruction = true.
   * After the call, @p pos points to the instruction originally at @p pos.
   */
  void add_invariant_assumes(
    goto_programt::targett &pos,
    const std::vector<expr2tc> &invariants);
};

#endif /* GOTO_PROGRAMS_GOTO_K_INDUCTION_H_ */
