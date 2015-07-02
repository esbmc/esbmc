/*
 * goto_unwind.h
 *
 *  Created on: Jun 3, 2015
 *      Author: mramalho
 */

#ifndef GOTO_PROGRAMS_GOTO_K_INDUCTION_H_
#define GOTO_PROGRAMS_GOTO_K_INDUCTION_H_

#include <std_types.h>
#include <hash_cont.h>

#include <message_stream.h>

#include "goto_functions.h"
#include "goto_loops.h"

void goto_k_induction(
  goto_functionst &goto_functions,
  contextt &_context,
  message_handlert &message_handler);

class goto_k_inductiont : public goto_loopst
{
public:
  goto_k_inductiont(
    goto_functiont &_goto_function,
    contextt &_context,
    message_handlert &_message_handler) :
    goto_loopst(
      _goto_function,
      _message_handler),
    context(_context),
    state_counter(0),
    state(struct_typet())
  {
    // unwind loops
    if(function_loops.size())
      goto_k_induction();
  }

protected:
  contextt &context;

  unsigned int state_counter;
  struct_typet state;

  void goto_k_induction();
  void convert_loop(loopst &loop);

  void fill_state(loopst &loop);
  void create_symbols();
  void make_nondet_assign(goto_programt::targett &loop_head);
  void init_k_indice(goto_programt::targett &loop_head);
  void update_state_vector(goto_programt::targett &loop_head);

  void copy(const codet &code,
    goto_program_instruction_typet type,
    goto_programt &dest);

};

#endif /* GOTO_PROGRAMS_GOTO_K_INDUCTION_H_ */
