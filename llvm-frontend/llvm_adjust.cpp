/*
 * llvmadjust.cpp
 *
 *  Created on: Aug 30, 2015
 *      Author: mramalho
 */

#include "llvm_adjust.h"

#include <std_code.h>
#include <expr_util.h>

#include <ansi-c/c_types.h>

llvm_adjust::llvm_adjust(contextt &_context)
  : context(_context),
    ns(namespacet(context))
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
    convert_exprt(*it);

    // All statements inside a function body must be an code
    // so convert any expression to a code_expressiont
    convert_expr_to_codet(*it);
  }
}

void llvm_adjust::convert_exprt(exprt& expr)
{
  Forall_operands(it, expr)
    convert_exprt(*it);

  if(expr.is_member())
    convert_member(to_member_expr(expr));
  else if(expr.id() == "+" || expr.id() == "-")
  {
    convert_pointer_arithmetic(expr.op0());
    convert_pointer_arithmetic(expr.op1());
  }
}

void llvm_adjust::convert_member(member_exprt& expr)
{
  exprt& base = expr.struct_op();
  if(base.type().is_pointer())
  {
    exprt deref("dereference");
    deref.type() = base.type().subtype();
    deref.move_to_operands(base);
    base.swap(deref);
  }
}

void llvm_adjust::convert_pointer_arithmetic(exprt& expr)
{
  if(expr.type().is_array())
  {
    typet new_type;
    const typet &expr_type=ns.follow(expr.type());

    if(expr_type.is_array())
    {
      new_type.id("pointer");
      new_type.subtype()=expr_type.subtype();
    }

    if(new_type != expr_type)
    {
      if(new_type.is_pointer() && expr_type.is_array())
      {
        exprt index_expr("index", expr_type.subtype());
        index_expr.reserve_operands(2);
        index_expr.move_to_operands(expr);
        index_expr.copy_to_operands(gen_zero(index_type()));
        expr=exprt("address_of", new_type);
        expr.move_to_operands(index_expr);
      }
    }
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
