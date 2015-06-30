/*
 * loopst.h
 *
 *  Created on: Jun 30, 2015
 *      Author: mramalho
 */

#ifndef GOTO_PROGRAMS_GOTO_LOOPS_H_
#define GOTO_PROGRAMS_GOTO_LOOPS_H_

#include <std_types.h>
#include <hash_cont.h>

#include <message_stream.h>

#include "goto_functions.h"

class goto_loopst : public message_streamt
{
public:
  goto_loopst(
    goto_functiont &_goto_functions,
    message_handlert &_message_handler) :
    message_streamt(_message_handler),
    goto_function(_goto_functions)
  {
    find_function_loops();
  }

  void find_function_loops();
  void output(std::ostream &out = std::cout);

protected:
  goto_functiont &goto_function;

  typedef std::map<goto_programt::targett, goto_programt> function_loopst;
  function_loopst function_loops;

  void create_function_loop(
    goto_programt::instructionst::iterator loop_head,
    goto_programt::instructionst::iterator loop_exit);
};

#endif /* GOTO_PROGRAMS_GOTO_LOOPS_H_ */
