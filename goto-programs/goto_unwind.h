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
  const namespacet &ns,
  message_handlert &message_handler);

class goto_unwindt:public message_streamt
{
public:
  goto_unwindt(
    goto_functiont &_goto_function,
    const namespacet &_ns,
    message_handlert &_message_handler):
    message_streamt(_message_handler),
    goto_function(_goto_function),
    ns(_ns)
  {
    // Find loops
    find_function_loops();

    // unwind loops
    goto_unwind_rec();
  }

protected:
  goto_functiont &goto_function;
  const namespacet &ns;

  typedef unsigned int loop_head;
  typedef unsigned int loop_exit;
  typedef std::map<loop_head, loop_exit> loops;

  loops function_loops;

  void find_function_loops();
  void create_function_loop(unsigned int loop_head, unsigned int loop_exit);

  void goto_unwind_rec();
};

#endif /* GOTO_PROGRAMS_GOTO_UNWIND_H_ */
