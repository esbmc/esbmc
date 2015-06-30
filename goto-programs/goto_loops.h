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

class loopst
{
public:
  loopst(goto_programt _goto_program) :
    goto_program(_goto_program)
  {}

  goto_programt& get_goto_program()
  {
    return goto_program;
  }

  void add_var_to_loop(const exprt &expr)
  {
    loop_vars.insert(
      std::pair<irep_idt, const exprt>(expr.identifier(), expr));
  }

protected:
  goto_programt goto_program;

  typedef std::map<irep_idt, const exprt> loop_varst;
  loop_varst loop_vars;
};

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

  typedef std::map<goto_programt::targett, loopst> function_loopst;
  function_loopst function_loops;

  void create_function_loop(
    goto_programt::instructionst::iterator loop_head,
    goto_programt::instructionst::iterator loop_exit);
};

#endif /* GOTO_PROGRAMS_GOTO_LOOPS_H_ */
