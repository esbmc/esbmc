/*
 * loopst.cpp
 *
 *  Created on: Jul 20, 2015
 *      Author: mramalho
 */

#include <goto-programs/loopst.h>

bool check_var_name(const exprt &expr)
{
  std::size_t found = expr.identifier().as_string().find("__ESBMC_");
  if(found != std::string::npos)
    return false;

  found = expr.identifier().as_string().find("__CPROVER");
  if(found != std::string::npos)
    return false;

  found = expr.identifier().as_string().find("return_value___");
  if(found != std::string::npos)
    return false;

  found = expr.identifier().as_string().find("c::pthread_lib");
  if(found != std::string::npos)
    return false;

  // Don't add variables that we created for k-induction
  found = expr.identifier().as_string().find("$");
  if(found != std::string::npos)
    return false;

  if(expr.identifier().as_string() == "c::__func__"
     || expr.identifier().as_string() == "c::__PRETTY_FUNCTION__"
     || expr.identifier().as_string() == "c::__LINE__")
    return false;

  if(expr.location().file().as_string() == "<built-in>"
     || expr.cmt_location().file().as_string() == "<built-in>"
     || expr.type().location().file().as_string() == "<built-in>"
     || expr.type().cmt_location().file().as_string() == "<built-in>")
    return false;

  return true;
}

loopst::loop_varst &loopst::get_loop_vars()
{
  return loop_vars;
}

goto_programt::targett loopst::get_original_loop_exit() const
{
  return original_loop_exit;
}

void loopst::set_original_loop_exit(goto_programt::targett _loop_exit)
{
  original_loop_exit = _loop_exit;
}

goto_programt::targett loopst::get_original_loop_head() const
{
  return original_loop_head;
}

void loopst::set_original_loop_head(goto_programt::targett _loop_head)
{
  original_loop_head = _loop_head;
}

void loopst::add_var_to_loop(loopst::loop_varst &_loop_vars)
{
  loop_vars.insert(_loop_vars.begin(), _loop_vars.end());
}

void loopst::add_var_to_loop(const exprt &expr)
{
  loop_vars.insert(
    std::pair<irep_idt, const exprt>(expr.identifier(), expr));
}

bool loopst::is_loop_var(exprt& expr)
{
  return (loop_vars.find(expr.identifier()) != loop_vars.end());
}

bool loopst::is_infinite_loop()
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

bool loopst::is_nondet_loop()
{
  goto_programt::targett tmp = original_loop_head;

  // First, check if the loop condition is a function
  // If it is a function, get the guard from the next instruction
  if(original_loop_head->is_assign())
  {
    ++tmp;
    exprt guard = migrate_expr_back(tmp->guard);
    assert(!guard.is_nil());
    return check_nondet(guard);
  }
  else
  {
    exprt guard = migrate_expr_back(tmp->guard);
    assert(!guard.is_nil());
    if(guard.is_not())
      return check_nondet(guard.op0());
    return check_nondet(guard);
  }

  return false;
}

bool loopst::check_nondet(exprt &guard)
{
  irep_idt exprid = guard.id();

  if(guard.is_false()
    || guard.is_constant()
    || guard.is_index())
  {
  }
  else if(exprid == ">"
    || exprid == ">="
    || guard.is_typecast())
  {
    if (!is_expr_a_constant(guard.op0()))
      return true;
  }
  else if(exprid == "<"
    || exprid == "<=")
  {
    if (!is_expr_a_constant(guard.op1()))
      return true;
  }
  else if (guard.is_and()
    || guard.is_or()
    || exprid == "-"
    || exprid == "+"
    || exprid == "*"
    || exprid == "/"
    || guard.is_notequal())
  {
    assert(guard.operands().size()==2);

    check_nondet(guard.op0());
    check_nondet(guard.op1());
  }
  else if (guard.is_dereference())
  {
    check_nondet(guard.op0());
  }
  else if (guard.is_symbol())
  {
    if (!is_expr_a_constant(guard))
      return true;
  }
  else
  {
    std::cerr << "**** Warning: this program " << guard.location().get_file()
              << " contains a '" << guard.id() << "' unsupported operator at line "
              << guard.location().get_line()
              << ", so we are not applying the inductive step to this program!"
              << std::endl;
    abort();
  }

  return false;
}

void loopst::dump()
{
  unsigned n=original_loop_head->location_number;

  std::cout << n << " is head of (size: ";
  std::cout << size;
  std::cout << ") { ";

  for(goto_programt::instructionst::iterator l_it = original_loop_head;
      l_it != original_loop_exit;
      ++l_it)
  {
    if(l_it != original_loop_head) std::cout << ", ";
    std::cout << (*l_it).location_number;
  }
  std::cout << " }" << std::endl;

  dump_loop_vars();
}

void loopst::dump_loop_vars()
{
  std::cout << "Loop variables:" << std::endl;

  u_int i = 0;
  for (std::pair<irep_idt, const exprt> expr : loop_vars)
    std::cout << ++i << ". \t" << "identifier: " << expr.first << std::endl
    << " " << expr.second << std::endl << std::endl;
  std::cout << std::endl;
}

bool loopst::is_expr_a_constant(exprt& expr)
{
  // TODO: handle pointers
  if (expr.is_constant() || expr.type().id() == "pointer")
    return true;

  if(expr.is_symbol())
  {
    exprt value = namespacet(context).lookup(expr.identifier()).value;
    if (value.is_constant())
      return true;
  }

  return false;
}
