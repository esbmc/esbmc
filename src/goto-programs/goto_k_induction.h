/*
 * goto_unwind.h
 *
 *  Created on: Jun 3, 2015
 *      Author: mramalho
 */

#ifndef GOTO_PROGRAMS_GOTO_K_INDUCTION_H_
#define GOTO_PROGRAMS_GOTO_K_INDUCTION_H_

#include <goto-programs/goto_functions.h>
#include <goto-programs/goto_loops.h>
#include <util/guard.h>
#include <util/hash_cont.h>
#include <util/message_stream.h>
#include <util/irep2_expr.h>

void goto_k_induction(
  goto_functionst &goto_functions,
  message_handlert &message_handler);

class goto_k_inductiont : public goto_loopst
{
public:
  goto_k_inductiont(
    const irep_idt &_function_name,
    goto_functionst &_goto_functions,
    goto_functiont &_goto_function,
    message_handlert &_message_handler)
    : goto_loopst(
        _function_name,
        _goto_functions,
        _goto_function,
        _message_handler)
  {
    if(function_loops.size())
      goto_k_induction();
  }

protected:
  typedef hash_map_cont<unsigned, bool> marked_branchst;
  marked_branchst marked_branch;

  typedef hash_map_cont<unsigned, guardt> guardst;
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
};

#endif /* GOTO_PROGRAMS_GOTO_K_INDUCTION_H_ */
