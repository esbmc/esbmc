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
#include <util/std_types.h>

void goto_k_induction(
  goto_functionst &goto_functions,
  contextt &_context,
  optionst &options,
  message_handlert &message_handler);

class goto_k_inductiont : public goto_loopst
{
public:
  goto_k_inductiont(
    const irep_idt &_function_name,
    goto_functionst &_goto_functions,
    goto_functiont &_goto_function,
    contextt &_context,
    optionst &_options,
    message_handlert &_message_handler) :
    goto_loopst(
      _context,
      _function_name,
      _goto_functions,
      _goto_function,
      _message_handler),
    state(struct_typet()),
    constrain_all_states(_options.get_bool_option("constrain-all-states")),
    options(_options)
  {
    // unwind loops
    if(function_loops.size())
      goto_k_induction();
  }

protected:
  struct_typet state;
  bool constrain_all_states;
  optionst &options;

  void goto_k_induction();

  void fill_state(loopst &loop);

  void convert_finite_loop(loopst &loop);

  void get_loop_cond(
    goto_programt::targett &loop_head,
    goto_programt::targett &loop_exit,
    exprt &loop_cond);

  void make_nondet_assign(goto_programt::targett &loop_head);

  void assume_loop_cond_before_loop(
    goto_programt::targett &loop_head,
    exprt &loop_cond);

  void assume_neg_loop_cond_after_loop(
    goto_programt::targett &loop_exit,
    exprt &loop_cond);

  void duplicate_loop_body(
    goto_programt::targett &loop_head,
    goto_programt::targett &loop_exit);

  void convert_assert_to_assume(
    goto_programt::targett &loop_head,
    goto_programt::targett &loop_exit);

  void adjust_loop_head_and_exit(
    goto_programt::targett &loop_head,
    goto_programt::targett &loop_exit);

  void copy(const codet &code,
    goto_program_instruction_typet type,
    goto_programt &dest);

  void assume_cond(
    const exprt &cond,
    goto_programt &dest);
};

#endif /* GOTO_PROGRAMS_GOTO_K_INDUCTION_H_ */
