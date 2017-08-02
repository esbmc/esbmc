/*******************************************************************\

Module: C++ Language Type Checking

Author: Daniel Kroening, kroening@kroening.com

\*******************************************************************/

#include <ansi-c/c_typecheck_base.h>
#include <util/expr_util.h>
#include <util/i2string.h>

void c_typecheck_baset::start_typecheck_code()
{
  case_is_allowed=break_is_allowed=continue_is_allowed=false;
}

void c_typecheck_baset::typecheck_code(codet &code)
{
  if(!code.is_code())
    throw "expected code, got "+code.pretty();

  code.type()=code_typet();

  const irep_idt &statement=code.statement();

//  std::cout << "statement: " << statement << std::endl;
//  std::cout << "typecheck_code::code.pretty(): " << code.pretty() << std::endl;

  if(statement=="expression")
    typecheck_expression(code);
  else if(statement=="label")
    typecheck_label(to_code_label(code));
  else if(statement=="switch_case")
    typecheck_switch_case(to_code_switch_case(code));
  else if(statement=="block")
    typecheck_block(code);
  else if(statement=="ifthenelse")
    typecheck_ifthenelse(code);
  else if(statement=="while" ||
          statement=="dowhile")
    typecheck_while(code);
  else if(statement=="for")
    typecheck_for(code);
  else if(statement=="switch")
    typecheck_switch(code);
  else if(statement=="break")
    typecheck_break(code);
  else if(statement=="goto")
    typecheck_goto(code);
  else if(statement=="continue")
    typecheck_continue(code);
  else if(statement=="return")
    typecheck_return(code);
  else if(statement=="decl")
    typecheck_decl(code);
  else if(statement=="assign")
    typecheck_assign(code);
  else if(statement=="skip")
  {
  }
  else if(statement=="asm")
    typecheck_asm(code);
  else if(statement=="start_thread")
    typecheck_start_thread(code);
  else if(statement=="msc_try_finally")
  {
    assert(code.operands().size()==2);
    typecheck_code(to_code(code.op0()));
    typecheck_code(to_code(code.op1()));
  }
  else if(statement=="msc_try_except")
  {
    assert(code.operands().size()==3);
    typecheck_code(to_code(code.op0()));
    typecheck_expr(code.op1());
    typecheck_code(to_code(code.op2()));
  }
  else if(statement=="msc_leave")
  {
    // fine as is, but should check that we
    // are in a 'try' block
  }
  else
  {
    err_location(code);
    str << "c_typecheck_baset: unexpected statement: " << statement;
    throw 0;
  }
}

void c_typecheck_baset::typecheck_asm(codet &code __attribute__((unused)))
{
}

void c_typecheck_baset::typecheck_assign(codet &code)
{
  if(code.operands().size()!=2)
    throw "assignment statement expected to have two operands";

  typecheck_expr(code.op0());
  typecheck_expr(code.op1());

  implicit_typecast(code.op1(), code.op0().type());
}

void c_typecheck_baset::typecheck_block(codet &code)
{
  Forall_operands(it, code)
    typecheck_code(to_code(*it));

  // do decl-blocks

  exprt new_ops;
  new_ops.operands().reserve(code.operands().size());

  Forall_operands(it1, code)
  {
    if(it1->is_nil()) continue;

    codet &code_op=to_code(*it1);

    if(code_op.get_statement()=="label")
    {
      // these may be nested
      codet *code_ptr=&code_op;

      while(code_ptr->get_statement()=="label")
      {
        assert(code_ptr->operands().size()==1);
        code_ptr=&to_code(code_ptr->op0());
      }

      new_ops.move_to_operands(code_op);
    }
    else
      new_ops.move_to_operands(code_op);
  }

  code.operands().swap(new_ops.operands());
}

void c_typecheck_baset::typecheck_break(codet &code)
{
  if(!break_is_allowed)
  {
    err_location(code);
    throw "break not allowed here";
  }
}

void c_typecheck_baset::typecheck_continue(codet &code)
{
  if(!continue_is_allowed)
  {
    err_location(code);
    throw "continue not allowed here";
  }
}

void c_typecheck_baset::typecheck_decl(codet &code)
{
  if(code.operands().size()!=1 &&
     code.operands().size()!=2)
  {
    err_location(code);
    throw "decl expected to have one or two arguments";
  }

  // op0 must be symbol
  if(code.op0().id()!="symbol")
  {
    err_location(code);
    throw "decl expected to have symbol as first operand";
  }

  // replace if needed
  replace_symbol(code.op0());

  // look it up
  const irep_idt &identifier=code.op0().identifier();

  symbolt* s = context.find_symbol(identifier);
  if(s == nullptr)
  {
    err_location(code);
    throw "failed to find decl symbol in context";
  }

  symbolt &symbol = *s;

  // see if it's a typedef
  // or a function
  // or static
  if(symbol.is_type ||
     symbol.type.is_code() ||
     symbol.static_lifetime)
  {
    locationt location=code.location();
    code=code_skipt();
    code.location()=location;
    return;
  }

  code.location()=symbol.location;

  // check the initializer, if any
  if(code.operands().size()==2)
  {
    typecheck_expr(code.op1());
    do_initializer(code.op1(), symbol.type, false);
  }

  // set type now (might be changed by initializer)
  code.op0().type()=symbol.type;

  const typet &type=follow(code.op0().type());

  // this must not be an incomplete type
  if(type.id()=="incomplete_struct" ||
     type.id()=="incomplete_array")
  {
    err_location(code);
    throw "incomplete type not permitted here";
  }
}

void c_typecheck_baset::typecheck_expression(codet &code)
{
  if(code.operands().size()!=1)
    throw "expression statement expected to have one operand";

  exprt &op=code.op0();
  typecheck_expr(op);

  if(op.id()=="sideeffect")
  {
    const irep_idt &statement=op.statement();

    if(statement=="assign")
    {
      assert(op.operands().size()==2);

      // pull assignment statements up
      exprt::operandst operands;
      operands.swap(op.operands());
      code.statement("assign");
      code.operands().swap(operands);

      if(code.op1().id()=="sideeffect" &&
         code.op1().statement()=="function_call")
      {
        assert(code.op1().operands().size()==2);

        code_function_callt function_call;
        function_call.location().swap(code.op1().location());
        function_call.lhs()=code.op0();
        function_call.function()=code.op1().op0();
        function_call.arguments()=code.op1().op1().operands();
        code.swap(function_call);
      }
    }
    else if(statement=="function_call")
    {
      assert(op.operands().size()==2);

      // pull function calls up
      code_function_callt function_call;
      function_call.location()=code.location();
      function_call.function()=op.op0();
      function_call.arguments()=op.op1().operands();
      code.swap(function_call);
    }
  }
}

void c_typecheck_baset::typecheck_for(codet &code)
{
  if(code.operands().size()!=4)
    throw "for expected to have four operands";

  // the "for" statement has an implicit block around it,
  // since code.op0() may contain declarations
  //
  // we therefore transform
  //
  //   for(a;b;c) d;
  //
  // to
  //
  //   { a; for(;b;c) d; }

  if(code.op0().is_not_nil())
    typecheck_code(to_code(code.op0()));

  if (code.op1().is_nil())
    code.op1().make_true();
  else
  {
    typecheck_expr(code.op1());
    implicit_typecast_bool(code.op1());
  }

  if (code.op2().is_not_nil())
    typecheck_code(to_code(code.op2()));

  if (code.op3().is_not_nil())
  {
    // save & set flags
    bool old_break_is_allowed(break_is_allowed);
    bool old_continue_is_allowed(continue_is_allowed);

    break_is_allowed = continue_is_allowed = true;

    typecheck_code(to_code(code.op3()));

    // restore flags
    break_is_allowed = old_break_is_allowed;
    continue_is_allowed = old_continue_is_allowed;
  }
}

void c_typecheck_baset::typecheck_label(code_labelt &code)
{
  if(code.operands().size()!=1)
  {
    err_location(code);
    throw "label expected to have one operand";
  }

  typecheck_code(to_code(code.op0()));
}

void c_typecheck_baset::typecheck_switch_case(code_switch_caset &code)
{
  if(code.operands().size()!=2)
  {
    err_location(code);
    throw "label expected to have one operand";
  }

  typecheck_code(code.code());

  if(code.is_default())
  {
    if(!case_is_allowed)
    {
      err_location(code);
      throw "did not expect default label here";
    }
  }
  else
  {
    if(!case_is_allowed)
    {
      err_location(code);
      throw "did not expect `case' here";
    }

    exprt &case_expr=code.case_op();
    typecheck_expr(case_expr);
    implicit_typecast(case_expr, switch_op_type);
  }
}

void c_typecheck_baset::typecheck_goto(codet &code __attribute__((unused)))
{
}

void c_typecheck_baset::typecheck_ifthenelse(codet &code)
{
  if(code.operands().size()!=2 &&
     code.operands().size()!=3)
    throw "ifthenelse expected to have two or three operands";

  exprt &cond=code.op0();

  typecheck_expr(cond);

  if(cond.id()=="sideeffect" &&
     cond.statement()=="assign")
  {
    err_location(cond);
    warning("warning: assignment in if condition");
  }

  implicit_typecast_bool(cond);

  typecheck_code(to_code(code.op1()));

  if(code.operands().size()==3 &&
     !code.op2().is_nil())
    typecheck_code(to_code(code.op2()));
}

void c_typecheck_baset::typecheck_start_thread(codet &code)
{
  if(code.operands().size()!=1)
    throw "start_thread expected to have one operand";

  typecheck_code(to_code(code.op0()));
}

void c_typecheck_baset::typecheck_return(codet &code)
{
  if(code.operands().size()==0)
  {
    if(return_type.id()!="empty")
    {
      err_location(code);
      throw "function expected to return a value";
    }
  }
  else if(code.operands().size()==1)
  {
    typecheck_expr(code.op0());

    if(return_type.id()=="empty")
    {
      if(code.op0().type().id()!="empty")
      {
        err_location(code);
        throw "function not expected to return a value";
      }
    }
    else
      implicit_typecast(code.op0(), return_type);
  }
  else
  {
    err_location(code);
    throw "return expected to have 0 or 1 operands";
  }
}

void c_typecheck_baset::typecheck_switch(codet &code)
{
  if(code.operands().size()!=2)
    throw "switch expects two operands";

  typecheck_expr(code.op0());

  // this needs to be promoted
  implicit_typecast_arithmetic(code.op0());

  // save & set flags

  bool old_case_is_allowed(case_is_allowed);
  bool old_break_is_allowed(break_is_allowed);
  typet old_switch_op_type(switch_op_type);

  switch_op_type=code.op0().type();
  break_is_allowed=case_is_allowed=true;

  typecheck_code(to_code(code.op1()));

  // restore flags
  case_is_allowed=old_case_is_allowed;
  break_is_allowed=old_break_is_allowed;
  switch_op_type=old_switch_op_type;
}

void c_typecheck_baset::typecheck_while(codet &code)
{
  if(code.operands().size()!=2)
    throw "while expected to have two operands";

  typecheck_expr(code.op0());
  implicit_typecast_bool(code.op0());

  // save & set flags
  bool old_break_is_allowed(break_is_allowed);
  bool old_continue_is_allowed(continue_is_allowed);

  break_is_allowed=continue_is_allowed=true;

  typecheck_code(to_code(code.op1()));

  // restore flags
  break_is_allowed=old_break_is_allowed;
  continue_is_allowed=old_continue_is_allowed;
}
