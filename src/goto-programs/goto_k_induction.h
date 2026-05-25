#ifndef GOTO_PROGRAMS_GOTO_K_INDUCTION_H_
#define GOTO_PROGRAMS_GOTO_K_INDUCTION_H_

#include <goto-programs/goto_functions.h>
#include <goto-programs/goto_loops.h>
#include <irep2/irep2_guard.h>
#include <irep2/irep2_expr.h>

void goto_k_induction(goto_functionst &goto_functions);

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
  typedef std::unordered_map<unsigned, guard2tc> guardst;
  guardst guards;

  /// Cached result of expanding a forward GOTO branch during the
  /// entry-condition collection. The boolean is the recursion's return
  /// value at this branch (`false_branch && true_branch`, i.e. true iff
  /// both subbranches reach the loop end); the guardst is the set of
  /// guards that should be merged into the caller's local guardst when
  /// this cache entry fires. Storing only the boolean (the legacy
  /// design) silently dropped these guards on every cache hit, weakening
  /// the entry-condition assume.
  struct branch_cache_entryt
  {
    bool reaches;
    guardst guards_to_merge;
  };
  typedef std::unordered_map<unsigned, branch_cache_entryt> marked_branchst;
  marked_branchst marked_branch;

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
};

#endif /* GOTO_PROGRAMS_GOTO_K_INDUCTION_H_ */
