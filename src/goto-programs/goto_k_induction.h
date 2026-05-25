#ifndef GOTO_PROGRAMS_GOTO_K_INDUCTION_H_
#define GOTO_PROGRAMS_GOTO_K_INDUCTION_H_

#include <goto-programs/goto_functions.h>
#include <goto-programs/goto_loop_transform.h>
#include <goto-programs/loopst.h>
#include <irep2/irep2_guard.h>
#include <irep2/irep2_expr.h>

void goto_k_induction(goto_functionst &goto_functions);

/// Per-loop k-induction transformation: havoc the loop's modified
/// variables and inject an ASSUME of the loop entry condition right
/// before the loop head. Inherits the function-and-loop iteration
/// scaffold from goto_loop_transformt.
class goto_k_inductiont : public goto_loop_transformt
{
public:
  explicit goto_k_inductiont(goto_functionst &_goto_functions)
    : goto_loop_transformt(_goto_functions)
  {
  }

  /// Apply the havoc + assume-entry-cond transformation to one loop.
  /// Public so other passes (e.g. goto_terminationt) can delegate to
  /// this method directly when iterating their own loops, without
  /// going through the full run() iteration.
  void transform_loop(
    const irep_idt &function_name,
    goto_functiont &goto_function,
    loopst &loop) override;

protected:
  typedef std::unordered_map<unsigned, guard2tc> guardst;

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

  /// Loop-scoped cache of `get_entry_cond_rec` results. Cleared at the
  /// start of every `transform_loop` so nested loops in the same
  /// function don't reuse stale entries.
  marked_branchst marked_branch;

  bool get_entry_cond_rec(
    const goto_programt::targett &loop_head,
    const goto_programt::targett &after_exit,
    guardst &guards);

  void make_nondet_assign(
    goto_functiont &goto_function,
    goto_programt::targett &loop_head,
    const loopst &loop);

  void remove_unrelated_loop_cond(guardst &guards, loopst &loop);

  void assume_loop_entry_cond_before_loop(
    goto_functiont &goto_function,
    goto_programt::targett &loop_head,
    const guardst &guards);

  void adjust_loop_head_and_exit(
    goto_programt::targett &loop_head,
    goto_programt::targett &loop_exit);
};

#endif /* GOTO_PROGRAMS_GOTO_K_INDUCTION_H_ */
