/*
 * clang_c_adjust.cpp
 *
 *  Created on: Aug 30, 2015
 *      Author: mramalho
 */

#include <std_code.h>
#include <expr_util.h>
#include <bitvector.h>
#include <prefix.h>
#include <cprover_prefix.h>
#include <c_types.h>

#include <ansi-c/c_sizeof.h>
#include "clang_c_adjust.h"

#include "typecast.h"

void clang_c_adjust::adjust_code(codet& code)
{
  const irep_idt &statement=code.statement();

  if(statement=="expression")
  {
    adjust_expression(code);
  }
  else if(statement=="block" ||
          statement=="decl-block")
  {
    adjust_blocks(code);
  }
  else if(statement=="ifthenelse")
  {
    adjust_ifthenelse(code);
  }
  else if(statement=="while" ||
          statement=="dowhile")
  {
    adjust_while(code);
  }
  else if(statement=="for")
  {
    adjust_for(code);
  }
  else if(statement=="switch")
  {
    adjust_switch(code);
  }
  else if(statement=="assign")
  {
    adjust_assign(code);
  }
  else if(statement=="decl")
  {
    adjust_decl(code);
  }
}

void clang_c_adjust::adjust_blocks(codet& code)
{
  Forall_operands(it, code)
    adjust_expr(*it);
}

void clang_c_adjust::adjust_expression(codet& code)
{
  exprt &op=code.op0();

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

void clang_c_adjust::adjust_decl(codet& code)
{
  if(code.operands().size() != 2)
    return;

  // Create typecast on assingments, if needed
  gen_typecast(ns, code.op1(), code.op0().type());
}

void clang_c_adjust::adjust_ifthenelse(codet& code)
{
  // If the condition is not of boolean type, it must be casted
  gen_typecast_bool(ns, code.op0());
}

void clang_c_adjust::adjust_while(codet& code)
{
  // If the condition is not of boolean type, it must be casted
  gen_typecast_bool(ns, code.op0());
}

void clang_c_adjust::adjust_for(codet& code)
{
  // If the condition is not of boolean type, it must be casted
  gen_typecast_bool(ns, code.op1());
}

void clang_c_adjust::adjust_switch(codet& code)
{
  // If the condition is not of int type, it must be casted
  gen_typecast_arithmetic(ns, code.op0());
}

void clang_c_adjust::adjust_assign(codet& code)
{
  // Create typecast on assingments, if needed
  gen_typecast(ns, code.op1(), code.op0().type());
}
