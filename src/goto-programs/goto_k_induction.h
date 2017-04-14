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
  void convert_infinite_loop(loopst &loop);
  void fill_state(loopst &loop);
  void create_symbols();
  void make_nondet_state_assign(goto_programt::targett &loop_head);
  void init_k_indice(goto_programt::targett &loop_head);
  void update_state_vector(goto_programt::targett &loop_head);
  void assign_current_state(goto_programt::targett &loop_exit);
  void assume_state_vector(goto_programt::targett &loop_exit);
  void assume_all_state_vector(goto_programt::targett &loop_exit);
  void convert_loop_body(loopst &loop);
  void convert_instruction(
    loopst &loop,
    goto_programt::targett loop_head,
    const irep_idt &_function_name);
  void replace_guard(loopst &loop, exprt &expr);
  void replace_by_cs_member(exprt &expr);
  void replace_ternary(
    loopst &loop,
    exprt &expr,
    bool is_if_cond = false);

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

  void kindice_incr(goto_programt &dest);

  symbol_exprt gen_kindice();
  symbol_exprt gen_state_vector();
  symbol_exprt gen_current_state();

  exprt gen_state_vector_indexed(exprt index);
};

#endif /* GOTO_PROGRAMS_GOTO_K_INDUCTION_H_ */
