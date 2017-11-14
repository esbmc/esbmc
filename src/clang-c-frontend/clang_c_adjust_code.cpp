/*
 * clang_c_adjust.cpp
 *
 *  Created on: Aug 30, 2015
 *      Author: mramalho
 */

#include <clang-c-frontend/clang_c_adjust.h>
#include <clang-c-frontend/typecast.h>
#include <util/bitvector.h>
#include <util/c_types.h>
#include <util/cprover_prefix.h>
#include <util/expr_util.h>
#include <util/prefix.h>
#include <util/std_code.h>

void clang_c_adjust::adjust_code(codet& code)
{
  const irep_idt &statement=code.statement();

  if(statement=="ifthenelse")
  {
    adjust_ifthenelse(code);
  }
  else if(statement=="while" || statement=="dowhile")
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
  else if(statement=="function_call")
  {
  }
  else
  {
    adjust_operands(code);
  }
}

void clang_c_adjust::adjust_decl(codet& code)
{
  if(code.operands().size() == 1) {
    adjust_type(code.op0().type());
    return;
  }

  assert(code.operands().size() == 2);

  // Check assignment
  adjust_expr(code.op1());

  // Check type
  adjust_type(code.op0().type());

  // Create typecast on assingments, if needed
  gen_typecast(ns, code.op1(), code.op0().type());
}

void clang_c_adjust::adjust_ifthenelse(codet& code)
{
  adjust_operands(code);

  // If the condition is not of boolean type, it must be casted
  gen_typecast_bool(ns, code.op0());
}

void clang_c_adjust::adjust_while(codet& code)
{
  adjust_operands(code);

  // If the condition is not of boolean type, it must be casted
  gen_typecast_bool(ns, code.op0());
}

void clang_c_adjust::adjust_for(codet& code)
{
  adjust_operands(code);

  // If the condition is not of boolean type, it must be casted
  gen_typecast_bool(ns, code.op1());
}

void clang_c_adjust::adjust_switch(codet& code)
{
  adjust_operands(code);

  // If the condition is not of int type, it must be casted
  gen_typecast_arithmetic(ns, code.op0());
}

void clang_c_adjust::adjust_assign(codet& code)
{
  adjust_operands(code);

  // Create typecast on assingments, if needed
  gen_typecast(ns, code.op1(), code.op0().type());
}
