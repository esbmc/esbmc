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

  typedef std::map<irep_idt, const exprt> loop_varst;
  loop_varst &get_loop_vars()
  {
    return loop_vars;
  }

  goto_programt::targett get_original_loop_exit() const
  {
    return original_loop_exit;
  }

  void set_original_loop_exit(goto_programt::targett _loop_exit)
  {
    original_loop_exit = _loop_exit;
  }

  goto_programt::targett get_original_loop_head() const
  {
    return original_loop_head;
  }

  void set_original_loop_head(goto_programt::targett _loop_head)
  {
    original_loop_head = _loop_head;
  }

  void add_var_to_loop(loop_varst &_loop_vars)
  {
    loop_vars.insert(_loop_vars.begin(), _loop_vars.end());
  }

  void add_var_to_loop(const exprt &expr)
  {
    loop_vars.insert(
      std::pair<irep_idt, const exprt>(expr.identifier(), expr));
  }

  void dump_loop_vars()
  {
    std::cout << "Loop variables:" << std::endl;

    u_int i = 0;
    for (std::pair<irep_idt, const exprt> expr : loop_vars)
      std::cout << ++i << ". \t" << "identifier: " << expr.first << std::endl
      << " " << expr.second << std::endl << std::endl;
    std::cout << std::endl;
  }

  bool is_loop_var(exprt& expr)
  {
    return (loop_vars.find(expr.identifier()) != loop_vars.end());
  }

  bool is_infinite_loop()
  {
    goto_programt::targett tmp = original_loop_head;

    // First, check if the loop condition is a function
    // If it is a function, get the guard from the next instruction
    if(original_loop_head->is_assign())
    {
      ++tmp;
      exprt guard = migrate_expr_back(tmp->guard);
      return guard.is_true();
    }
    else
    {
      exprt guard = migrate_expr_back(tmp->guard);
      assert(!guard.is_nil());

      if(guard.is_true())
        return true;

      if(guard.is_not())
        return guard.op0().is_true();

      return false;
    }

    return false;
  }

  void output(std::ostream &out = std::cout)
  {
    unsigned n=original_loop_head->location_number;

    out << n << " is head of { ";

    for(goto_programt::instructionst::iterator l_it=
        goto_program.instructions.begin();
        l_it != goto_program.instructions.end();
        ++l_it)
    {
      if(l_it != goto_program.instructions.begin()) out << ", ";
        out << (*l_it).location_number;
    }
    out << " }\n";

    dump_loop_vars();
  }

protected:
  goto_programt goto_program;
  loop_varst loop_vars;

  goto_programt::targett original_loop_head;
  goto_programt::targett original_loop_exit;
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

  void add_loop_var(loopst &loop, const exprt &expr);
};

#endif /* GOTO_PROGRAMS_GOTO_LOOPS_H_ */
