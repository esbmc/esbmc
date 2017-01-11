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
  if(code.operands().size() != 2)
    return;

  // Check assignment
  adjust_expr(code.op1());

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
