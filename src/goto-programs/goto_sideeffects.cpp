/*******************************************************************\

Module: Program Transformation

Author: Daniel Kroening, kroening@kroening.com

\*******************************************************************/

#include <expr_util.h>
#include <std_expr.h>
#include <rename.h>
#include <cprover_prefix.h>
#include <i2string.h>
#include <c_types.h>

#include "goto_convert_class.h"

void goto_convertt::make_temp_symbol(
  exprt &expr,
  goto_programt &dest)
{
  const locationt location=expr.find_location();

  symbolt &new_symbol=new_tmp_symbol(expr.type());

  code_assignt assignment;
  assignment.lhs()=symbol_expr(new_symbol);
  assignment.rhs()=expr;
  assignment.location()=location;

  convert(assignment, dest);

  expr=symbol_expr(new_symbol);
}

void goto_convertt::read(exprt &expr, goto_programt &dest)
{
  if(expr.is_constant())
    return;

  if(expr.id()=="symbol")
  {
    // see if we already renamed it

  }

  symbolt &new_symbol=new_tmp_symbol(expr.type());

  codet assignment("assign");
  assignment.reserve_operands(2);
  assignment.copy_to_operands(symbol_expr(new_symbol));
  assignment.move_to_operands(expr);

  goto_programt tmp_program;
  convert(assignment, tmp_program);

  dest.destructive_append(tmp_program);

  expr=symbol_expr(new_symbol);
}

bool goto_convertt::has_sideeffect(const exprt &expr)
{
  forall_operands(it, expr)
    if(has_sideeffect(*it))
      return true;

  if(expr.id()=="sideeffect")
    return true;

  return false;
}

bool goto_convertt::has_function_call(const exprt &expr)
{
  forall_operands(it, expr)
    if(has_function_call(*it))
      return true;

  if(expr.id()=="sideeffect" &&
     expr.statement()=="function_call")
    return true;

  return false;
}

void goto_convertt::remove_sideeffects(
  exprt &expr,
  goto_programt &dest,
  bool result_is_used)
{
  guardt guard;
  remove_sideeffects(expr, guard, dest, result_is_used);
}

void goto_convertt::remove_sideeffects(
  exprt &expr,
  guardt &guard,
  goto_programt &dest,
  bool result_is_used)
{

  if(!has_sideeffect(expr))
    return;

  if(expr.is_and() || expr.id()=="or")
  {
    if(!expr.is_boolean())
      throw expr.id_string()+" must be Boolean, but got "+
            expr.pretty();

    std::vector<bool> s;
    s.resize(expr.operands().size());
    unsigned last=0;

    // Ben : see if operands have sizeeffects as well and remember it if there is
    for(unsigned i=0; i<expr.operands().size(); i++)
    {
      s[i]=has_sideeffect(expr.operands()[i]);
      if(s[i]) last=i;
    }

    guardt old_guards(guard);

    for(unsigned i=0; i<=last; i++)
    {
      exprt &op=expr.operands()[i];

      if(!op.is_boolean())
        throw expr.id_string()+" takes Boolean operands only, but got "+
              op.pretty();

      if(s[i])
      {
        // the side effect might modify the previous conditions
        for(unsigned j=0; j<i; j++)
          read(expr.operands()[j], dest);

        remove_sideeffects(op, guard, dest);
      }

      if(expr.id()=="or")
      {
        exprt tmp(op);
        tmp.make_not();
        expr2tc tmp_expr;
        migrate_expr(tmp, tmp_expr);
        guard.add(tmp_expr);
      }
      else
      {
        expr2tc tmp_expr;
        migrate_expr(op, tmp_expr);
        guard.add(tmp_expr);
      }
    }

    guard.swap(old_guards);

    return;
  }
  else if(expr.id()=="if")
  {
    if(expr.operands().size()!=3)
      throw "if takes three arguments";

    if(!expr.op0().is_boolean())
    {
      std::string msg=
        "first argument of if must be boolean, but got "
        +expr.op0().to_string();
      throw msg;
    }

    remove_sideeffects(expr.op0(), guard, dest);

    bool o1=has_sideeffect(expr.op1());
    bool o2=has_sideeffect(expr.op2());

    if(o1 || o2)
      read(expr.op0(), dest);

    if(o1)
    {
      guardt old_guards(guard);
      expr2tc tmp;
      migrate_expr(expr.op0(), tmp);
      guard.add(tmp);
      remove_sideeffects(expr.op1(), guard, dest);
      guard.swap(old_guards);
    }

    if(o2)
    {
      guardt old_guards(guard);
      exprt tmp(expr.op0());
      tmp.make_not();
      expr2tc tmp_expr;
      migrate_expr(tmp, tmp_expr);
      guard.add(tmp_expr);
      remove_sideeffects(expr.op2(), guard, dest);
      guard.swap(old_guards);
    }

    return;
  }
  else if(expr.id()=="comma")
  {
    exprt result;

    Forall_operands(it, expr)
    {
      bool last=(it==--expr.operands().end());

      if(last)
      {
        result.swap(*it);
        remove_sideeffects(result, guard, dest, result_is_used);
      }
      else
        remove_sideeffects(*it, guard, dest, false);
    }

    expr.swap(result);

    return;
  }
  else if(expr.id()=="typecast")
  {
    if(expr.operands().size()!=1)
      throw "typecast takes one argument";

    // preserve 'result_is_used'
    remove_sideeffects(expr.op0(), guard, dest, result_is_used);

    return;
  }
  else if(expr.id()=="sideeffect" &&
          expr.statement()=="gcc_conditional_expression")
  {
    remove_gcc_conditional_expression(expr, guard, dest);
    return;
  }

  // TODO: evaluation order
  Forall_operands(it, expr)
    remove_sideeffects(*it, guard, dest);

  if(expr.id()=="sideeffect")
  {
    const irep_idt &statement=expr.statement();

    if(statement=="function_call") // might do anything
      remove_function_call(expr, guard, dest, result_is_used);
    else if(statement=="assign" ||
            statement=="assign+" ||
            statement=="assign-" ||
            statement=="assign*" ||
            statement=="assign_div" ||
            statement=="assign_bitor" ||
            statement=="assign_bitxor" ||
            statement=="assign_bitand" ||
            statement=="assign_lshr" ||
            statement=="assign_ashr" ||
            statement=="assign_shl" ||
            statement=="assign_mod")
      remove_assignment(expr, guard, dest);
    else if(statement=="postincrement" ||
            statement=="postdecrement")
      remove_post(expr, guard, dest, result_is_used);
    else if(statement=="preincrement" ||
            statement=="predecrement")
      remove_pre(expr, guard, dest);
    else if(statement=="cpp_new" ||
            statement=="cpp_new[]")
      remove_cpp_new(expr, guard, dest, result_is_used);
    else if(statement=="temporary_object")
      remove_temporary_object(expr, guard, dest, result_is_used);
    else if(statement=="statement_expression")
      remove_statement_expression(expr, guard, dest, result_is_used);
    else if(statement=="nondet")
    {
      // these are fine
    }
    else if(statement=="typeid")
    {
      // Let's handle typeid later (goto_function.cpp, do_function_call)
    }
    else
    {
      str << "cannot remove side effect (" << statement << ")";
      throw 0;
    }
  }
}

void goto_convertt::address_of_replace_objects(
  exprt &expr,
  goto_programt &dest)
{
  if(expr.id()=="struct" ||
     expr.id()=="union" ||
     expr.is_array())
  {
    make_temp_symbol(expr, dest);
    return;
  }
  else if(expr.id()=="string-constant")
  {
  }
  else
    Forall_operands(it, expr)
      address_of_replace_objects(*it, dest);
}

void goto_convertt::remove_assignment(
  exprt &expr,
  guardt &guard,
  goto_programt &dest)
{
  codet assignment_statement("expression");
  assignment_statement.copy_to_operands(expr);
  assignment_statement.location()=expr.location();

  if(expr.operands().size()!=2)
    throw "assignment must have two operands";

  exprt lhs;
  lhs.swap(expr.op0());
  expr.swap(lhs);

  goto_programt tmp_program;
  convert(assignment_statement, tmp_program);
  guard_program(guard, tmp_program);
  dest.destructive_append(tmp_program);
}

void goto_convertt::remove_pre(
  exprt &expr,
  guardt &guard,
  goto_programt &dest)
{
  codet pre_statement("expression");
  pre_statement.copy_to_operands(expr);

  if(expr.operands().size()!=1)
    throw "preincrement/predecrement must have one operand";

  exprt op;
  op.swap(expr.op0());
  expr.swap(op);

  goto_programt tmp_program;
  convert(pre_statement, tmp_program);
  guard_program(guard, tmp_program);
  dest.destructive_append(tmp_program);
}

void goto_convertt::remove_post(
  exprt &expr,
  guardt &guard,
  goto_programt &dest,
  bool result_is_used)
{
  // we have ...(op++)...
  codet post_statement("expression");
  post_statement.copy_to_operands(expr);

  if(expr.operands().size()!=1)
    throw "postincrement/postdecrement must have one operand";

  if(result_is_used)
  {
    exprt tmp;
    tmp.swap(expr.op0());
    read(tmp, dest);
    expr.swap(tmp);
  }
  else
  {
    expr.make_nil();
  }

  goto_programt tmp_program;
  convert(post_statement, tmp_program);
  guard_program(guard, tmp_program);
  dest.destructive_append(tmp_program);
}

void goto_convertt::remove_function_call(
  exprt &expr,
  guardt &guard,
  goto_programt &dest,
  bool result_is_used)
{
  codet call;

  if(result_is_used)
  {
    symbolt new_symbol;

    new_symbol.base_name="return_value";
    new_symbol.lvalue=true;
    new_symbol.type=expr.type();

    // get name of function, if available

    if(expr.id()!="sideeffect" ||
       expr.statement()!="function_call")
      throw "expected function call";

    if(expr.operands().empty())
      throw "function_call expects at least one operand";

    if(expr.op0().id()=="symbol")
    {
      const irep_idt &identifier=expr.op0().identifier();
      const symbolt &symbol=ns.lookup(identifier);

      std::string new_base_name=id2string(new_symbol.base_name);

      new_base_name+="_";
      new_base_name+=id2string(symbol.base_name);
      new_base_name+="$"+i2string(++temporary_counter);

      new_symbol.base_name=new_base_name;
      new_symbol.mode=symbol.mode;
    }

    new_symbol.name=tmp_symbol_prefix+id2string(new_symbol.base_name);

    new_name(new_symbol);

    tmp_symbols.push_back(new_symbol.name);

    call=code_assignt(symbol_expr(new_symbol), expr);

    expr=symbol_expr(new_symbol);
  }
  else
  {
    call=codet("expression");
    call.move_to_operands(expr);
  }

  goto_programt tmp_program;
  convert(call, tmp_program);
  guard_program(guard, tmp_program);
  dest.destructive_append(tmp_program);
}

void goto_convertt::replace_new_object(
  const exprt &object,
  exprt &dest)
{
  if(dest.id()=="new_object")
    dest=object;
  else
    Forall_operands(it, dest)
      replace_new_object(object, *it);
}

void goto_convertt::remove_cpp_new(
  exprt &expr,
  guardt &guard,
  goto_programt &dest,
  bool result_is_used)
{
  codet call;

  if(result_is_used)
  {
    symbolt new_symbol;

    new_symbol.base_name="new_value$"+i2string(++temporary_counter);
    new_symbol.lvalue=true;
    new_symbol.type=expr.type();
    new_symbol.name=tmp_symbol_prefix+id2string(new_symbol.base_name);

    new_name(new_symbol);
    tmp_symbols.push_back(new_symbol.name);

    call=code_assignt(symbol_expr(new_symbol), expr);

    expr=symbol_expr(new_symbol);
  }
  else
  {
    call=codet("expression");
    call.move_to_operands(expr);
  }

  goto_programt tmp_program;
  convert(call, tmp_program);
  guard_program(guard, tmp_program);
  dest.destructive_append(tmp_program);
}

void goto_convertt::remove_temporary_object(
  exprt &expr,
  guardt &guard __attribute__((unused)),
  goto_programt &dest,
  bool result_is_used __attribute__((unused)))
{
  if(expr.operands().size()!=1 &&
     expr.operands().size()!=0)
    throw "temporary_object takes 0 or 1 operands";

  symbolt &new_symbol=new_tmp_symbol(expr.type());

  new_symbol.mode=expr.mode();

  if(expr.operands().size()==1)
  {
    codet assignment("assign");
    assignment.reserve_operands(2);
    assignment.copy_to_operands(symbol_expr(new_symbol));
    assignment.move_to_operands(expr.op0());

    goto_programt tmp_program;
    convert(assignment, tmp_program);
    dest.destructive_append(tmp_program);
  }

  if(expr.initializer().is_not_nil())
  {
    assert(expr.operands().empty());
    exprt initializer=static_cast<const exprt &>(expr.initializer());
    replace_new_object(symbol_expr(new_symbol), initializer);

    goto_programt tmp_program;
    convert(to_code(initializer), tmp_program);
    dest.destructive_append(tmp_program);
  }

  expr=symbol_expr(new_symbol);
}

void goto_convertt::remove_statement_expression(
  exprt &expr,
  guardt &guard,
  goto_programt &dest,
  bool result_is_used)
{
  if(expr.operands().size()!=1)
    throw "statement_expression takes 1 operand";

  if(!expr.op0().is_code())
    throw "statement_expression takes code as operand";

  codet &code=to_code(expr.op0());

  exprt last;
  last.make_nil();

  if(result_is_used)
  {
    // get last statement from block
    if(code.get_statement()!="block")
      throw "statement_expression expects block";

    if(code.operands().empty())
      throw "statement_expression expects non-empty block";

    last.swap(code.operands().back());
    code.operands().pop_back();
  }

  {
    goto_programt tmp;
    convert(code, tmp);
    guard_program(guard, tmp);

    dest.destructive_append(tmp);
  }

  if(result_is_used)
  {
    goto_programt tmp;
    remove_sideeffects(last, guard, tmp, true);

    if(last.statement()=="expression" &&
       last.operands().size()==1)
      expr=last.op0();
    else
      throw "statement_expression expects expression as last statement";
  }
}

void goto_convertt::remove_gcc_conditional_expression(
  exprt &expr,
  guardt &guard,
  goto_programt &dest)
{
  if(expr.operands().size()!=2)
    throw "conditional_expression takes two operands";

  // first remove side-effects from condition
  remove_sideeffects(expr.op0(), guard, dest);

  exprt if_expr("if");
  if_expr.operands().resize(3);

  if_expr.op0()=expr.op0();
  if_expr.op1()=expr.op0();
  if_expr.op2()=expr.op1();
  if_expr.location()=expr.location();

  if(if_expr.op0().type()!=bool_typet())
    if_expr.op0().make_typecast(bool_typet());

  expr.swap(if_expr);

  // there might still be one in expr.op2()
  remove_sideeffects(expr, guard, dest);
}
