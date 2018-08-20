/*
 * loopst.h
 *
 *  Created on: Jun 30, 2015
 *      Author: mramalho
 */

#ifndef GOTO_PROGRAMS_GOTO_LOOPS_H_
#define GOTO_PROGRAMS_GOTO_LOOPS_H_

#include <goto-programs/goto_functions.h>
#include <goto-programs/loopst.h>
#include <util/hash_cont.h>
#include <util/message_stream.h>
#include <util/std_types.h>

class goto_loopst : public message_streamt
{
protected:
  const irep_idt function_name;
  goto_functionst &goto_functions;
  goto_functiont &goto_function;

  typedef std::list<loopst> function_loopst;
  function_loopst function_loops;

  void create_function_loop(
    goto_programt::instructionst::iterator loop_head,
    goto_programt::instructionst::iterator loop_exit);

  void get_modified_variables(
    goto_programt::instructionst::iterator instruction,
    function_loopst::iterator loop,
    const irep_idt &_function_name);

  void add_modified_var(loopst &loop, const expr2tc &expr);
  void add_unmodified_var(loopst &loop, const expr2tc &expr);

  void add_loop_var(loopst &loop, const expr2tc &expr, bool is_modified);

public:
  goto_loopst(
    const irep_idt &_function_name,
    goto_functionst &_goto_functions,
    goto_functiont &_goto_function,
    message_handlert &_message_handler)
    : message_streamt(_message_handler),
      function_name(_function_name),
      goto_functions(_goto_functions),
      goto_function(_goto_function)
  {
    find_function_loops();
  }

  void find_function_loops();
  void dump() const;

  const function_loopst &get_loops() const
  {
    return function_loops;
  }
};

#endif /* GOTO_PROGRAMS_GOTO_LOOPS_H_ */
