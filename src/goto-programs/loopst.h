#ifndef GOTO_PROGRAMS_LOOPST_H_
#define GOTO_PROGRAMS_LOOPST_H_

#include <goto-programs/goto_functions.h>
#include <unordered_set>

class loopst
{
public:
  loopst() : size(0)
  {
  }

  typedef std::unordered_set<expr2tc, irep2_hash> loop_varst;

  const loop_varst &get_modified_loop_vars() const;
  const loop_varst &get_unmodified_loop_vars() const;

  const goto_programt::targett get_original_loop_exit() const;
  void set_original_loop_exit(goto_programt::targett _loop_exit);

  const goto_programt::targett get_original_loop_head() const;
  void set_original_loop_head(goto_programt::targett _loop_head);

  void add_modified_var_to_loop(const expr2tc &expr);
  void add_unmodified_var_to_loop(const expr2tc &expr);

  void dump() const;
  void dump_loop_vars() const;
  void output_to(std::ostream &oss) const;
  void output_loop_vars_to(std::ostream &oss) const;

  void set_size(std::size_t size)
  {
    this->size = size;
  }

protected:
  loop_varst modified_loop_vars;
  loop_varst unmodified_loop_vars;

  goto_programt::targett original_loop_head;
  goto_programt::targett original_loop_exit;

  std::size_t size;
};

#endif /* GOTO_PROGRAMS_LOOPST_H_ */
