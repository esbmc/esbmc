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
  loopst(contextt &_context) : context(_context), size(0)
  {
  }

  typedef hash_set_cont<expr2tc, irep2_hash> loop_varst;

  const loop_varst &get_loop_vars() const;

  goto_programt::targett get_original_loop_exit() const;
  void set_original_loop_exit(goto_programt::targett _loop_exit);

  goto_programt::targett get_original_loop_head() const;
  void set_original_loop_head(goto_programt::targett _loop_head);

  void add_var_to_loop(loop_varst &_loop_vars);
  void add_var_to_loop(const expr2tc &expr);

  void dump();
  void dump_loop_vars();

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
};

#endif /* GOTO_PROGRAMS_LOOPST_H_ */
