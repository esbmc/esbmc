/*******************************************************************\

Module: Dereferencing Operations on GOTO Programs

Author: Daniel Kroening, kroening@kroening.com

\*******************************************************************/

#include <pointer-analysis/goto_program_dereference.h>
#include <util/base_type.h>
#include <util/irep2.h>
#include <util/migrate.h>
#include <util/prefix.h>
#include <util/simplify_expr.h>
#include <util/std_code.h>

bool goto_program_dereferencet::has_failed_symbol(
  const expr2tc &expr,
  const symbolt *&symbol)
{
  if (is_symbol2t(expr))
  {
    if (has_prefix(to_symbol2t(expr).thename.as_string(), "symex::invalid_object"))
      return false;

    // Null and invalid name lookups will fail.
    if (to_symbol2t(expr).thename == "NULL" ||
        to_symbol2t(expr).thename == "INVALID")
      return false;

    const symbolt &ptr_symbol = ns.lookup(migrate_expr_back(expr));

    const irep_idt &failed_symbol = ptr_symbol.type.failed_symbol();

    if (failed_symbol == "")
      return false;

    return !ns.lookup(failed_symbol, symbol);
  }

  return false;
}

bool goto_program_dereferencet::is_valid_object(
  const irep_idt &identifier)
{
  const symbolt &symbol=ns.lookup(identifier);

  if(symbol.type.is_code())
    return true;

  if(symbol.static_lifetime)
    return true; // global/static

  auto it = std::find(
    valid_local_variables->begin(),
    valid_local_variables->end(),
    symbol.name);

  if(it != valid_local_variables->end())
    return true; // valid local

  return false;
}

void goto_program_dereferencet::dereference_failure(
  const std::string &property,
  const std::string &msg,
  const guardt &guard)
{
  expr2tc guard_expr = guard.as_expr();

  if (assertions.insert(guard_expr).second)
  {
    guard_expr = not2tc(guard_expr);

    // first try simplifier on it
    if (!options.get_bool_option("no-simplify"))
    {
      base_type(guard_expr, ns);
      simplify(guard_expr);
    }

    if (!is_true(guard_expr))
    {
      goto_programt::targett t=new_code.add_instruction(ASSERT);
      t->guard = guard_expr;
      t->location=dereference_location;
      t->location.property(property);
      t->location.comment("dereference failure: "+msg);
    }
  }
}

void goto_program_dereferencet::get_value_set(
  const expr2tc &expr,
  value_setst::valuest &dest)
{
  value_sets.get_values(current_target, expr, dest);
}

void goto_program_dereferencet::dereference_expr(
  expr2tc &expr,
  const bool checks_only,
  const dereferencet::modet mode)
{
  guardt guard;

  if(checks_only) {
    expr2tc tmp = expr;
    dereference.dereference_expr(tmp, guard, mode);
  } else {
    dereference.dereference_expr(expr, guard, mode);
  }
}

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

    valid_local_variables = &goto_program.local_variables;
    dereference_instruction(it, checks_only);

    // insert new instructions
    while(!new_code.instructions.empty())
    {
      goto_program.insert_swap(it, new_code.instructions.front());
      new_code.instructions.pop_front();
      it++;
    }
  }

  goto_program.local_variables.insert(
    goto_program.local_variables.begin(),
    new_code.local_variables.begin(),
    new_code.local_variables.end());
}

void goto_program_dereferencet::dereference_program(
  goto_functionst &goto_functions,
  bool checks_only)
{
  for(auto & it : goto_functions.function_map)
    dereference_program(it.second.body, checks_only);
}

void goto_program_dereferencet::dereference_instruction(
  goto_programt::targett target,
  bool checks_only)
{
  current_target=target;
  goto_programt::instructiont &i=*target;
  dereference_location = i.location;

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

    if (!is_nil_expr(func_call.ret))
      dereference_expr(func_call.ret, checks_only, dereferencet::WRITE);

    for (auto &it : func_call.operands)
      dereference_expr(it, checks_only, dereferencet::READ);

    if (is_dereference2t(func_call.function)) {
      // Rather than derefing function ptr, which we're moving to not collect
      // via pointer analysis, instead just assert that it's a valid pointer.
      const dereference2t &deref = to_dereference2t(func_call.function);
      invalid_pointer2tc invalid_ptr(deref.value);
      guardt guard;
      guard.add(invalid_ptr);
      if(!options.get_bool_option("no-pointer-check"))
      {
        dereference_failure("function pointer dereference",
                            "invalid pointer", guard);
      }
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
      i.code->Foreach_operand([this, &checks_only] (expr2tc &e) {
          dereference_expr(e, checks_only, dereferencet::READ);
        }
      );
    } else if (is_code_free2t(i.code)) {
      code_free2t &free = to_code_free2t(i.code);
      expr2tc operand = free.operand;

      guardt guard;
      // Result discarded
      dereference.dereference(operand, operand->type, guard, dereferencet::FREE,
                              expr2tc());
    }
  }
}

void goto_program_dereferencet::dereference_expression(
  goto_programt::const_targett target,
  expr2tc &expr)
{
  current_target = target;
  dereference_expr(expr, false, dereferencet::READ);
}

void goto_program_dereferencet::pointer_checks(
  goto_programt &goto_program)
{
  dereference_program(goto_program, true);
}

void goto_program_dereferencet::pointer_checks(
  goto_functionst &goto_functions)
{
  dereference_program(goto_functions, true);
}

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

void pointer_checks(
  goto_functionst &goto_functions,
  const namespacet &ns,
  contextt &context,
  const optionst &options,
  value_setst &value_sets)
{
  goto_program_dereferencet
    goto_program_dereference(ns, context, options, value_sets);
  goto_program_dereference.pointer_checks(goto_functions);
}

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
