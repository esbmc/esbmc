/*
 * goto_unwind.h
 *
 *  Created on: Jun 3, 2015
 *      Author: mramalho
 */

#ifndef GOTO_PROGRAMS_GOTO_UNWIND_H_
#define GOTO_PROGRAMS_GOTO_UNWIND_H_

#include <std_types.h>
#include <hash_cont.h>

#include <message_stream.h>

#include "goto_functions.h"

void goto_unwind(
  goto_functionst &goto_functions,
  unsigned unwind,
  const namespacet &ns,
  message_handlert &message_handler);

class goto_unwindt:public message_streamt
{
public:
  goto_unwindt(
    goto_functiont &_goto_function,
    unsigned _unwind,
    const namespacet &_ns,
    message_handlert &_message_handler) :
    message_streamt(_message_handler),
    goto_function(_goto_function),
    unwind(_unwind),
    ns(_ns),
    tmp_goto_program(goto_programt())
  {
    // Find loops
    find_function_loops();

    // unwind loops
    if(function_loops.size())
      goto_unwind();
  }

protected:
  goto_functiont &goto_function;
  unsigned unwind;
  const namespacet &ns;

  goto_programt tmp_goto_program;

  typedef std::map<unsigned, goto_programt> function_loopst;
  function_loopst function_loops;

  void find_function_loops();
  void create_function_loop(
    goto_programt::instructionst::iterator loop_head,
    goto_programt::instructionst::iterator loop_exit);

  void handle_nested_loops();
  void handle_nested_loops_rec(
    function_loopst::iterator,
    bool rec);

  void goto_unwind();

};

#endif /* GOTO_PROGRAMS_GOTO_UNWIND_H_ */
