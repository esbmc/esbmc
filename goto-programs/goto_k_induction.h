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

void goto_k_induction(
  goto_functionst &goto_functions,
  const namespacet &ns,
  message_handlert &message_handler);

class goto_k_inductiont:public message_streamt
{
public:
  goto_k_inductiont(
    goto_functiont &_goto_function,
    const namespacet &_ns,
    message_handlert &_message_handler) :
    message_streamt(_message_handler),
    goto_function(_goto_function),
    ns(_ns)
  {
  }

protected:
  goto_functiont &goto_function;
  const namespacet &ns;

  typedef std::map<goto_programt::targett, goto_programt> function_loopst;
  function_loopst function_loops;

  void output(std::ostream &out);

  void find_function_loops();
  void create_function_loop(
    goto_programt::instructionst::iterator loop_head,
    goto_programt::instructionst::iterator loop_exit);
};

#endif /* GOTO_PROGRAMS_GOTO_K_INDUCTION_H_ */
