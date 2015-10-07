/*
 * llvmadjust.cpp
 *
 *  Created on: Aug 30, 2015
 *      Author: mramalho
 */

#include "llvm_adjust.h"

#include <std_code.h>
#include <expr_util.h>
#include <bitvector.h>
#include <prefix.h>
#include <cprover_prefix.h>

#include <ansi-c/c_types.h>
#include <ansi-c/c_sizeof.h>

#include "typecast.h"

void llvm_adjust::convert_code(codet& code)
{
  const irep_idt &statement=code.statement();

  if(statement=="expression")
  {
    convert_expression(code);
  }
  else if(statement=="label")
  {
    // We just need to convert the first statement inside the label
    // since any other statement will be converted by the
    // if(statement=="block")
    convert_expr_to_codet(code.op0());
  }
  else if(statement=="ifthenelse")
  {
    // If the condition is not of boolean type, it must be casted
    gen_typecast(code.op0(), bool_type());

    // Convert exprt when there is no block defined for ifthenelse
    convert_expr_to_codet(code.op1());
    if(code.operands().size() == 3)
      convert_expr_to_codet(code.op2());
  }
  else if(statement=="while" ||
          statement=="dowhile")
  {
    // If the condition is not of boolean type, it must be casted
    gen_typecast(code.op0(), bool_type());

    // Convert exprt when there is no block defined
    convert_expr_to_codet(code.op1());
  }
  else if(statement=="for")
  {
    // If the condition is not of boolean type, it must be casted
    gen_typecast(code.op1(), bool_type());

    // Convert exprt when there is no block defined
    convert_expr_to_codet(code.op0());
    convert_expr_to_codet(code.op2());
    convert_expr_to_codet(code.op3());
  }
  else if(statement=="switch")
  {
  }
  else if(statement=="assign")
  {
    // Creat typecast on assingments, if needed
    gen_typecast(code.op1(), code.op0().type());
  }
  else if(statement=="skip")
  {
  }
  else if(statement=="msc_try_finally")
  {
  }
  else if(statement=="msc_try_except")
  {
  }
  else if(statement=="msc_leave")
  {
  }
}

void llvm_adjust::convert_expression(codet& code)
{
  if(code.operands().size()!=1)
    throw "expression statement expected to have one operand";

  exprt &op=code.op0();
  convert_expr(op);

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
