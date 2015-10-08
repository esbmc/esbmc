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
    convert_expression(code);
  else if(statement=="label")
    convert_label(to_code_label(code));
  else if(statement=="block" ||
          statement=="decl-block")
    convert_block(code);
  else if(statement=="ifthenelse")
    convert_ifthenelse(code);
  else if(statement=="while" ||
          statement=="dowhile")
    convert_while(code);
  else if(statement=="for")
    convert_for(code);
  else if(statement=="switch")
    convert_switch(code);
  else if(statement=="assign")
    convert_assign(code);
  else if(statement=="return")
  {
  }
  else if(statement=="break")
  {
  }
  else if(statement=="goto")
  {
  }
  else if(statement=="continue")
  {
  }
  else if(statement=="decl")
  {
  }
  else if(statement=="skip")
  {
  }
  else if(statement=="asm")
  {
  }
  else if(statement=="function_call")
  {
  }
  else
  {
    std::cout << "Unexpected statement: " << statement << std::endl;
    code.dump();
    abort();
  }
}

void llvm_adjust::convert_expression(codet& code)
{
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

void llvm_adjust::convert_label(code_labelt& code)
{
  convert_code(to_code(code.op0()));

  if(code.case_irep().is_not_nil())
  {
    exprt case_expr=static_cast<const exprt &>(code.case_irep());

    Forall_operands(it, case_expr)
      convert_expr(*it);

    code.case_irep(case_expr);
  }
}

void llvm_adjust::convert_block(codet& code)
{
  Forall_operands(it, code)
    convert_code(to_code(*it));
}

void llvm_adjust::convert_ifthenelse(codet& code)
{
  exprt &cond=code.op0();
  convert_expr(cond);

  // If the condition is not of boolean type, it must be casted
  gen_typecast(ns, code.op0(), bool_type());

  convert_code(to_code(code.op1()));

  if(code.operands().size()==3 && !code.op2().is_nil())
    convert_code(to_code(code.op2()));
}

void llvm_adjust::convert_while(codet& code)
{
  // If the condition is not of boolean type, it must be casted
  gen_typecast(ns, code.op0(), bool_type());

  convert_expr(code.op0());
  convert_code(to_code(code.op1()));
}

void llvm_adjust::convert_for(codet& code)
{
  convert_code(to_code(code.op0()));
  convert_expr(code.op1());
  convert_code(to_code(code.op2()));
  convert_code(to_code(code.op3()));

  // If the condition is not of boolean type, it must be casted
  gen_typecast(ns, code.op1(), bool_type());
}

void llvm_adjust::convert_switch(codet& code)
{
  // If the condition is not of boolean type, it must be casted
  gen_typecast(ns, code.op0(), int_type());

  convert_expr(code.op0());
  convert_code(to_code(code.op1()));
}

void llvm_adjust::convert_assign(codet& code)
{
  convert_expr(code.op0());
  convert_expr(code.op1());

  // Create typecast on assingments, if needed
  gen_typecast(ns, code.op1(), code.op0().type());
}
