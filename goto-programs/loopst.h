/*
 * loopst.h
 *
 *  Created on: Jul 20, 2015
 *      Author: mramalho
 */

#ifndef GOTO_PROGRAMS_LOOPST_H_
#define GOTO_PROGRAMS_LOOPST_H_

#include "goto_functions.h"

class loopst
{
public:
  loopst(contextt &_context, goto_programt _goto_program) :
    context(_context),
    goto_program(_goto_program)
  {}

  typedef std::map<irep_idt, const exprt> loop_varst;

  goto_programt& get_goto_program();
  loop_varst &get_loop_vars();

  goto_programt::targett get_original_loop_exit() const;
  void set_original_loop_exit(goto_programt::targett _loop_exit);

  goto_programt::targett get_original_loop_head() const;
  void set_original_loop_head(goto_programt::targett _loop_head);

  void add_var_to_loop(loop_varst &_loop_vars);
  void add_var_to_loop(const exprt &expr);

  bool is_loop_var(exprt& expr);
  bool is_infinite_loop();
  bool is_nondet_loop();

  void output(std::ostream &out = std::cout);
  void dump_loop_vars();

protected:
  contextt &context;
  goto_programt goto_program;
  loop_varst loop_vars;

  goto_programt::targett original_loop_head;
  goto_programt::targett original_loop_exit;
};


#endif /* GOTO_PROGRAMS_LOOPST_H_ */
