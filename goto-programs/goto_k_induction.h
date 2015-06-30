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
  message_handlert &message_handler);

class goto_k_inductiont : public goto_loopst
{
public:
  goto_k_inductiont(
    goto_functiont &_goto_function,
    message_handlert &_message_handler) :
    goto_loopst(
      _goto_function,
      _message_handler),
    state_counter(0),
    state(struct_typet())
  {
    // unwind loops
    if(function_loops.size())
      goto_k_induction();
  }

protected:
  typedef std::map<irep_idt, const exprt> loop_varst;
  loop_varst loop_vars;

  unsigned int state_counter;
  struct_typet state;

  void goto_k_induction();

};

#endif /* GOTO_PROGRAMS_GOTO_K_INDUCTION_H_ */
