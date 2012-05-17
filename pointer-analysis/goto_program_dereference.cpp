/*******************************************************************\

Module: Dereferencing Operations on GOTO Programs

Author: Daniel Kroening, kroening@kroening.com

\*******************************************************************/

#include <irep2.h>
#include <migrate.h>
#include <prefix.h>
#include <simplify_expr.h>
#include <base_type.h>
#include <std_code.h>

#include "goto_program_dereference.h"

/*******************************************************************\

Function: goto_program_dereferencet::has_failed_symbol

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

bool goto_program_dereferencet::has_failed_symbol(
  const expr2tc &expr,
  const symbolt *&symbol)
{
  if (is_symbol2t(expr))
  {
    if (has_prefix(to_symbol2t(expr).name.as_string(), "symex::invalid_object"))
      return false;

    exprt tmp_sym = migrate_expr_back(expr);
    const symbolt &ptr_symbol = ns.lookup(tmp_sym);

    const irep_idt &failed_symbol = ptr_symbol.type.failed_symbol();

    if (failed_symbol == "")
      return false;

    return !ns.lookup(failed_symbol, symbol);
  }

  return false;
}

/*******************************************************************\

Function: goto_program_dereferencet::is_valid_object

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

bool goto_program_dereferencet::is_valid_object(
  const irep_idt &identifier)
{
  const symbolt &symbol=ns.lookup(identifier);

  if(symbol.type.is_code())
    return true;

  if(symbol.static_lifetime)
    return true; // global/static

  if(valid_local_variables->find(symbol.name)!=
     valid_local_variables->end())
    return true; // valid local

  return false;
}

/*******************************************************************\

Function: goto_program_dereferencet::dereference_failure

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void goto_program_dereferencet::dereference_failure(
  const std::string &property,
  const std::string &msg,
  const guardt &guard)
{
  exprt tmp_guard_expr = guard.as_expr();
  expr2tc guard_expr;
  migrate_expr(tmp_guard_expr, guard_expr);

  if (assertions.insert(guard_expr).second)
  {
    guard_expr = expr2tc(new not2t(guard_expr));

    // first try simplifier on it
    if (!options.get_bool_option("no-simplify"))
    {
      base_type(guard_expr, ns);
      expr2tc tmp_expr = guard_expr->simplify();
      if (!is_nil_expr(tmp_expr))
        guard_expr = tmp_expr;
    }

    if (!is_constant_bool2t(guard_expr) ||
        !to_constant_bool2t(guard_expr).constant_value)
    {
      goto_programt::targett t=new_code.add_instruction(ASSERT);
      t->guard = guard_expr;
      t->location=dereference_location;
      t->location.property(property);
      t->location.comment("dereference failure: "+msg);
    }
  }
}

/*******************************************************************\

Function: goto_program_dereferencet::dereference_rec

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void goto_program_dereferencet::dereference_rec(
  expr2tc &expr,
  guardt &guard,
  const dereferencet::modet mode)
{

  if (!dereference.has_dereference(expr))
    return;

  if (is_and2t(expr) || is_or2t(expr))
  {
    unsigned old_guards=guard.size();

    assert(is_bool_type(expr->type));

    std::vector<expr2tc *> operands;
    expr.get()->list_operands(operands);
    for (unsigned i = 0; i < operands.size(); i++)
    {
      expr2tc &op = *operands[i];

      assert(is_bool_type(op->type));

      if (dereference.has_dereference(op))
        dereference_rec(op, guard, dereferencet::READ);

      if (is_or2t(expr)) {
        expr2tc tmp = expr2tc(new not2t(op));
        exprt tmp_expr = migrate_expr_back(tmp);
        guard.move(tmp_expr);
      } else {
        exprt tmp_expr = migrate_expr_back(op);
        guard.add(tmp_expr);
      }
    }

    guard.resize(old_guards);

    return;
  }
  else if (is_if2t(expr))
  {
    if2t &ifref = to_if2t(expr);
    assert(is_bool_type(ifref.cond->type));
    dereference_rec(ifref.cond, guard, dereferencet::READ);

    bool o1 = dereference.has_dereference(ifref.true_value);
    bool o2 = dereference.has_dereference(ifref.false_value);

    if (o1) {
      unsigned old_guard=guard.size();
      guard.add(migrate_expr_back(ifref.cond));
      dereference_rec(ifref.true_value, guard, mode);
      guard.resize(old_guard);
    }

    if (o2) {
      unsigned old_guard=guard.size();
      expr2tc tmp = expr2tc(new not2t(ifref.cond));
      exprt tmp_expr = migrate_expr_back(tmp);
      guard.move(tmp_expr);
      dereference_rec(ifref.false_value, guard, mode);
      guard.resize(old_guard);
    }

    return;
  }

  if (is_address_of2t(expr))
  {
    // turn &*p to p
    // this has *no* side effect!

    address_of2t &addrof = to_address_of2t(expr);

    if (is_dereference2t(addrof.ptr_obj)) {
      dereference2t &deref = to_dereference2t(addrof.ptr_obj);
      expr2tc result = deref.value;

      if (result->type != expr->type)
        result = expr2tc(new typecast2t(expr->type, result));

      expr = result;
    }
  }

  std::vector<expr2tc*> operands;
  expr.get()->list_operands(operands);
  for (std::vector<expr2tc*>::const_iterator it = operands.begin();
       it != operands.end(); it++)
    dereference_rec(**it, guard, mode);

  if (is_dereference2t(expr)) {
    dereference2t &deref = to_dereference2t(expr);

    expr2tc tmp_obj = deref.value;
    dereference.dereference(tmp_obj, guard, mode);
    expr = tmp_obj;
  } else if (is_index2t(expr)) {
    index2t &idx = to_index2t(expr);

    if (is_pointer_type(idx.source_value->type)) {
      expr2tc tmp = expr2tc(new add2t(idx.source_value->type, idx.source_value,
                                      idx.index));
      dereference.dereference(tmp, guard, mode);
    }
  }
}

/*******************************************************************\

Function: goto_program_dereferencet::get_value_set

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void goto_program_dereferencet::get_value_set(
  const expr2tc &expr,
  value_setst::valuest &dest)
{
  value_sets.get_values(current_target, expr, dest);
}

/*******************************************************************\

Function: goto_program_dereferencet::dereference_expr

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void goto_program_dereferencet::dereference_expr(
  expr2tc &expr,
  const bool checks_only,
  const dereferencet::modet mode)
{
  guardt guard;

  if(checks_only) {
    expr2tc tmp = expr;
    dereference_rec(expr, guard, mode);
  } else {
    dereference_rec(expr, guard, mode);
  }
}

/*******************************************************************\

Function: goto_program_dereferencet::dereference

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void goto_program_dereferencet::dereference_program(
  goto_programt &goto_program,
  bool checks_only)
{
  for(goto_programt::instructionst::iterator
      it=goto_program.instructions.begin();
      it!=goto_program.instructions.end();
      it++)
  {
    new_code.clear();
    assertions.clear();

    dereference_instruction(it, checks_only);

    for(goto_programt::instructionst::iterator
        i_it=new_code.instructions.begin();
        i_it!=new_code.instructions.end();
        i_it++)
      i_it->local_variables=it->local_variables;

    // insert new instructions
    while(!new_code.instructions.empty())
    {
      goto_program.insert_swap(it, new_code.instructions.front());
      new_code.instructions.pop_front();
      it++;
    }
  }
}

/*******************************************************************\

Function: goto_program_dereferencet::dereference

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void goto_program_dereferencet::dereference_program(
  goto_functionst &goto_functions,
  bool checks_only)
{
  for(goto_functionst::function_mapt::iterator
      it=goto_functions.function_map.begin();
      it!=goto_functions.function_map.end();
      it++)
    dereference_program(it->second.body, checks_only);
}

/*******************************************************************\

Function: goto_program_dereferencet::dereference

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void goto_program_dereferencet::dereference_instruction(
  goto_programt::targett target,
  bool checks_only)
{
  current_target=target;
  valid_local_variables=&target->local_variables;
  goto_programt::instructiont &i=*target;

  dereference_expr(i.guard, checks_only, dereferencet::READ);

  if (i.is_assign())
  {
    code_assign2t &assign = to_code_assign2t(i.code);
    dereference_expr(assign.target, checks_only, dereferencet::WRITE);
    dereference_expr(assign.source, checks_only, dereferencet::READ);
  }
  else if (i.is_function_call())
  {
    code_function_call2t &func_call = to_code_function_call2t(i.code);

    if (!is_nil_expr(func_call.ret)) {
      dereference_expr(func_call.ret, checks_only, dereferencet::WRITE);
    }

    for (std::vector<expr2tc>::iterator it = func_call.operands.begin();
         it != func_call.operands.end(); it++)
      dereference_expr(*it, checks_only, dereferencet::READ);

    if (is_dereference2t(func_call.function)) {
      // Rather than derefing function ptr, which we're moving to not collect
      // via pointer analysis, instead just assert that it's a valid pointer.
      const dereference2t &deref = to_dereference2t(func_call.function);
      expr2tc invalid_ptr = expr2tc(new invalid_pointer2t(deref.value));
      guardt guard;
      exprt tmp = migrate_expr_back(invalid_ptr);
      guard.move(tmp);
      dereference_failure("function pointer dereference",
                          "invalid pointer", guard);
    }
  }
  else if (i.is_return())
  {
    code_return2t &ret = to_code_return2t(i.code);
    if (is_nil_expr(ret.operand))
      return;

    dereference_expr(ret.operand, checks_only, dereferencet::READ);
  }
  else if(i.is_other())
  {
    if (is_code_decl2t(i.code)) {
      ;
    } else if (is_code_expression2t(i.code)) {
      code_expression2t &theexp = to_code_expression2t(i.code);
      dereference_expr(theexp.operand, checks_only, dereferencet::READ);
    } else if (is_code_printf2t(i.code)) {
      std::vector<expr2tc *> operands;
      i.code.get()->list_operands(operands);
      for (std::vector<expr2tc*>::const_iterator it = operands.begin();
           it != operands.end(); it++)
        dereference_expr(**it, checks_only, dereferencet::READ);
    } else if (is_code_free2t(i.code)) {
      code_free2t &free = to_code_free2t(i.code);
      expr2tc operand = free.operand;

      guardt guard;
      dereference.dereference(operand, guard, dereferencet::FREE);
    }
  }
}

/*******************************************************************\

Function: goto_program_dereferencet::dereference

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void goto_program_dereferencet::dereference_expression(
  goto_programt::const_targett target,
  expr2tc &expr)
{
  current_target=target;
  valid_local_variables=&target->local_variables;

  dereference_expr(expr, false, dereferencet::READ);
}

/*******************************************************************\

Function: goto_program_dereferencet::pointer_checks

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void goto_program_dereferencet::pointer_checks(
  goto_programt &goto_program)
{
  dereference_program(goto_program, true);
}

/*******************************************************************\

Function: goto_program_dereferencet::pointer_checks

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void goto_program_dereferencet::pointer_checks(
  goto_functionst &goto_functions)
{
  dereference_program(goto_functions, true);
}

/*******************************************************************\

Function: remove_pointers

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void remove_pointers(
  goto_programt &goto_program,
  contextt &context,
  const optionst &options,
  value_setst &value_sets)
{
  namespacet ns(context);

  goto_program_dereferencet
    goto_program_dereference(ns, context, options, value_sets);

  goto_program_dereference.dereference_program(goto_program);
}

/*******************************************************************\

Function: remove_pointers

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void remove_pointers(
  goto_functionst &goto_functions,
  contextt &context,
  const optionst &options,
  value_setst &value_sets)
{
  namespacet ns(context);

  goto_program_dereferencet
    goto_program_dereference(ns, context, options, value_sets);

  Forall_goto_functions(it, goto_functions)
    goto_program_dereference.dereference_program(it->second.body);
}

/*******************************************************************\

Function: pointer_checks

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void pointer_checks(
  goto_programt &goto_program,
  const namespacet &ns,
  const optionst &options,
  value_setst &value_sets)
{
  contextt new_context;
  goto_program_dereferencet
    goto_program_dereference(ns, new_context, options, value_sets);
  goto_program_dereference.pointer_checks(goto_program);
}

/*******************************************************************\

Function: pointer_checks

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void pointer_checks(
  goto_functionst &goto_functions,
  const namespacet &ns,
  const optionst &options,
  value_setst &value_sets)
{
  contextt new_context;
  goto_program_dereferencet
    goto_program_dereference(ns, new_context, options, value_sets);
  goto_program_dereference.pointer_checks(goto_functions);
}

/*******************************************************************\

Function: dereference

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void dereference(
  goto_programt::const_targett target,
  expr2tc &expr,
  const namespacet &ns,
  value_setst &value_sets)
{
  optionst options;
  contextt new_context;
  goto_program_dereferencet
    goto_program_dereference(ns, new_context, options, value_sets);

  goto_program_dereference.dereference_expression(target, expr);
}
