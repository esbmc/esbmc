/*
 * loopst.h
 *
 *  Created on: Jul 20, 2015
 *      Author: mramalho
 */

#ifndef GOTO_PROGRAMS_LOOPST_H_
#define GOTO_PROGRAMS_LOOPST_H_

#include <goto-programs/goto_functions.h>

class loopst
{
public:
  loopst(contextt &_context) :
    context(_context)
  {}

  typedef std::map<irep_idt, const exprt> loop_varst;

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

  void dump();
  void dump_loop_vars();

  std::size_t get_size() const
  {
    return size;
  }

  void set_size(std::size_t size)
  {
    this->size = size;
  }

protected:
  contextt &context;
  loop_varst loop_vars;

  goto_programt::targett original_loop_head;
  goto_programt::targett original_loop_exit;

  std::size_t size;

private:
  bool check_nondet(exprt &guard);
  bool is_expr_a_constant(exprt &expr);
};

bool check_var_name(const exprt &expr);

#endif /* GOTO_PROGRAMS_LOOPST_H_ */
