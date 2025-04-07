#include <goto-programs/loopst.h>

const loopst::loop_varst &loopst::get_modified_loop_vars() const
{
  return modified_loop_vars;
}

const loopst::loop_varst &loopst::get_unmodified_loop_vars() const
{
  return unmodified_loop_vars;
}

const goto_programt::targett loopst::get_original_loop_exit() const
{
  return original_loop_exit;
}

void loopst::set_original_loop_exit(goto_programt::targett _loop_exit)
{
  original_loop_exit = _loop_exit;
}

const goto_programt::targett loopst::get_original_loop_head() const
{
  return original_loop_head;
}

void loopst::set_original_loop_head(goto_programt::targett _loop_head)
{
  original_loop_head = _loop_head;
}

void loopst::add_modified_var_to_loop(const expr2tc &expr)
{
  modified_loop_vars.insert(expr);
}

void loopst::add_unmodified_var_to_loop(const expr2tc &expr)
{
  unmodified_loop_vars.insert(expr);
}
void loopst::output_to(std::ostream &oss) const
{
  unsigned n = get_original_loop_head()->location_number;

  oss << n << " is head of (size: ";
  oss << size;
  oss << ") { ";

  for (goto_programt::instructionst::iterator l_it = get_original_loop_head();
       l_it != get_original_loop_exit();
       ++l_it)
    oss << (*l_it).location_number << ", ";
  oss << get_original_loop_exit()->location_number;

  oss << " }"
      << "\n";
}

void loopst::output_loop_vars_to(std::ostream &oss) const
{
  oss << "Loop variables:\n";
  unsigned int i = 0;
  for (auto var : modified_loop_vars)
    oss << ++i << ". \t" << to_symbol2t(var).thename << '\n';
  oss << '\n';
}

void loopst::dump() const
{
  std::ostringstream oss;
  output_to(oss);
  log_status("{}", oss.str());
  dump_loop_vars();
}

void loopst::dump_loop_vars() const
{
  std::ostringstream oss;
  output_loop_vars_to(oss);
  log_status("{}", oss.str());
}

void loopst::skip_loop()
{
  goto_programt::targett loop_head = get_original_loop_head();
  goto_programt::targett loop_exit = get_original_loop_exit();
  while (loop_head != loop_exit)
  {
    loop_head->make_skip();
    loop_head++;
  }
  loop_exit->make_skip();
}
