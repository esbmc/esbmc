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
#include <util/hash_cont.h>
#include <util/message_stream.h>
#include <util/irep2_expr.h>

void goto_k_induction(
  goto_functionst &goto_functions,
  contextt &_context,
  message_handlert &message_handler);

class goto_k_inductiont : public goto_loopst
{
public:
  goto_k_inductiont(
    const irep_idt &_function_name,
    goto_functionst &_goto_functions,
    goto_functiont &_goto_function,
    contextt &_context,
    message_handlert &_message_handler) :
    goto_loopst(
      _context,
      _function_name,
      _goto_functions,
      _goto_function,
      _message_handler)
  {
    // unwind loops
    if(function_loops.size())
      goto_k_induction();
  }

protected:
  void goto_k_induction();

  void convert_finite_loop(loopst &loop);

  const expr2tc get_loop_cond(
    goto_programt::targett &loop_head,
    goto_programt::targett &loop_exit);

  void make_nondet_assign(
    goto_programt::targett &loop_head,
    const loopst &loop);

  void assume_loop_cond_before_loop(
    goto_programt::targett &loop_head,
    expr2tc &loop_cond);

  void assume_neg_loop_cond_after_loop(
    goto_programt::targett &loop_exit,
    expr2tc &loop_cond);

  void duplicate_loop_body(
    goto_programt::targett &loop_head,
    goto_programt::targett &loop_exit);

  void convert_assert_to_assume(
    goto_programt::targett &loop_head,
    goto_programt::targett &loop_exit);

  void adjust_loop_head_and_exit(
    goto_programt::targett &loop_head,
    goto_programt::targett &loop_exit);

  void assume_cond(const expr2tc &cond, goto_programt &dest);
};

#endif /* GOTO_PROGRAMS_GOTO_K_INDUCTION_H_ */
