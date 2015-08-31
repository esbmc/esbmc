/*
 * llvmadjust.cpp
 *
 *  Created on: Aug 30, 2015
 *      Author: mramalho
 */

#include "llvm_adjust.h"

#include <std_code.h>

llvm_adjust::llvm_adjust(contextt &_context)
  : context(_context)
{
}

llvm_adjust::~llvm_adjust()
{
}

bool llvm_adjust::adjust()
{
  Forall_symbols(it, context.symbols)
  {
    if(!it->second.is_type && it->second.type.is_code())
      adjust_function(it->second);
  }
  return false;
}

void llvm_adjust::adjust_function(symbolt& symbol)
{
  Forall_operands(it, symbol.value)
  {
    // All statements inside a function body must be an code
    // so convert any expression to a code_expressiont
    convert_expr_to_codet(*it);
  }
}

void llvm_adjust::convert_expr_to_codet(exprt& expr)
{
  if(expr.is_code())
    return;

  codet code("expression");
  code.copy_to_operands(expr);

  expr.swap(code);
}
