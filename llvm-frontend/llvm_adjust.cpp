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

#include <ansi-c/c_types.h>
#include <ansi-c/c_sizeof.h>

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
  {
    convert_member(to_member_expr(expr));
  }
  else if(expr.id() == "+" || expr.id() == "-")
  {
    convert_pointer_arithmetic(expr.op0());
    convert_pointer_arithmetic(expr.op1());
  }
  else if(expr.is_index())
  {
    convert_index(to_index_expr(expr));
  }
  else if(expr.is_dereference())
  {
    convert_dereference(expr);
  }
  else if(expr.id() == "sizeof")
  {
    convert_sizeof(expr);
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

void llvm_adjust::convert_index(index_exprt& index)
{
  exprt &array_expr=index.op0();
  exprt &index_expr=index.op1();

  // we might have to swap them

  {
    const typet &array_full_type=ns.follow(array_expr.type());
    const typet &index_full_type=ns.follow(index_expr.type());

    if(!array_full_type.is_array() &&
        array_full_type.id()!="incomplete_array" &&
        array_full_type.id()!="pointer" &&
        (index_full_type.is_array() ||
            index_full_type.id()=="incomplete_array" ||
            index_full_type.id()=="pointer"))
      std::swap(array_expr, index_expr);
  }

  make_index_type(index_expr);

  const typet &final_array_type=ns.follow(array_expr.type());

  if(final_array_type.is_array() ||
      final_array_type.id()=="incomplete_array")
  {
    if(array_expr.cmt_lvalue())
      index.cmt_lvalue(true);
  }
  else if(final_array_type.id()=="pointer")
  {
    // p[i] is syntactic sugar for *(p+i)

    exprt addition("+", array_expr.type());
    addition.operands().swap(index.operands());
    index.move_to_operands(addition);
    index.id("dereference");
    index.cmt_lvalue(true);
  }

  index.type()=final_array_type.subtype();
}

void llvm_adjust::convert_dereference(exprt& deref)
{
  exprt &op=deref.op0();

  const typet op_type=ns.follow(op.type());

  if(op_type.is_array() ||
     op_type.id()=="incomplete_array")
  {
    // *a is the same as a[0]
    deref.id("index");
    deref.type()=op_type.subtype();
    deref.copy_to_operands(gen_zero(index_type()));
    assert(deref.operands().size()==2);
  }
  else if(op_type.id()=="pointer")
  {
    if(op_type.subtype().id()=="empty")
    {
      std::cout << "operand of unary * is a void * pointer" << std::endl;
      abort();
    }

    deref.type()=op_type.subtype();
  }
  else
  {
    std::cout  << "operand of unary * `" << op.name().as_string()
        << "' is not a pointer";
    throw 0;
  }

  deref.cmt_lvalue(true);

  // if you dereference a pointer pointing to
  // a function, you get a pointer again
  // allowing ******...*p

  convert_expr_function_identifier(deref);
}

void llvm_adjust::convert_expr_to_codet(exprt& expr)
{
  if(expr.is_code())
    return;

  codet code("expression");
  code.copy_to_operands(expr);

  expr.swap(code);
}

void llvm_adjust::convert_sizeof(exprt& expr)
{
  typet type;

  if(expr.operands().size()==0)
  {
    type = ((typet &)expr.sizeof_type());
  }
  else if(expr.operands().size()==1)
  {
    type.swap(expr.op0().type());
  }
  else
  {
    std::cout << "sizeof operator expects zero or one operand, "
              << "but got" << expr.operands().size() << std::endl;
    abort();
  }

  exprt new_expr=c_sizeof(type, ns);

  if(new_expr.is_nil())
  {
    std::cout << "type has no size, " << type.name() << std::endl;
    abort();
  }

  new_expr.swap(expr);
  expr.cmt_c_sizeof_type(type);
}

void llvm_adjust::adjust_type(typet &type)
{
  if(type.id()=="symbol")
  {
    const irep_idt &identifier=type.identifier();

    symbolst::const_iterator s_it=context.symbols.find(identifier);

    if(s_it==context.symbols.end())
    {
      std::cout << "type symbol `" << identifier << "' not found" << std::endl;
      abort();
    }

    const symbolt &symbol=s_it->second;

    if(!symbol.is_type)
    {
      std::cout << "expected type symbol, but got " << std::endl;
      symbol.dump();
      abort();
    }

    if(symbol.is_macro)
      type=symbol.type; // overwrite
  }
}

void llvm_adjust::make_index_type(exprt& expr)
{
  const typet &full_type=ns.follow(expr.type());

  if(full_type.is_bool())
  {
    expr.make_typecast(index_type());
  }
  else if(full_type.id()=="unsignedbv")
  {
    unsigned width=bv_width(expr.type());

    if(width!=config.ansi_c.int_width)
      expr.make_typecast(uint_type());
  }
  else if(full_type.id()=="signedbv" ||
          full_type.id()=="c_enum" ||
          full_type.id()=="incomplete_c_enum")
  {
    if(full_type!=index_type())
      expr.make_typecast(index_type());
  }
  else
  {
    std::cout << "expected integer type, but got `"
        << full_type.name().as_string() << "'";
    throw 0;
  }
}

void llvm_adjust::convert_expr_function_identifier(exprt& expr)
{
  if(expr.type().is_code())
  {
    exprt tmp("address_of", pointer_typet());
    tmp.implicit(true);
    tmp.type().subtype()=expr.type();
    tmp.location()=expr.location();
    tmp.move_to_operands(expr);
    expr.swap(tmp);
  }
}
