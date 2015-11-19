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

bool llvm_adjust::adjust()
{
  Forall_symbols(it, context.symbols)
  {
    if(!it->second.is_type && it->second.type.is_code())
    {
      adjust_function(it->second);
    }
    else if(has_prefix(it->second.name.as_string(), CPROVER_PREFIX))
    {
      convert_builtin(it->second);
    }
  }
  return false;
}


void llvm_adjust::convert_builtin(symbolt& symbol)
{
  const irep_idt &identifier = symbol.name;

  // TODO: find a better solution for this
  if(has_prefix(id2string(identifier), CPROVER_PREFIX "alloc")
     || has_prefix(id2string(identifier), CPROVER_PREFIX "deallocated")
     || has_prefix(id2string(identifier), CPROVER_PREFIX "is_dynamic")
     || has_prefix(id2string(identifier), CPROVER_PREFIX "alloc_size"))
  {
    exprt expr=exprt("infinity", symbol.type.subtype());

    symbol.type.size(expr);
    symbol.value.type().size(expr);
  }
}

void llvm_adjust::adjust_function(symbolt& symbol)
{
  Forall_operands(it, symbol.value)
  {
    convert_expr(*it);
  }
}

void llvm_adjust::convert_expr(exprt& expr)
{
  if(expr.id()=="sideeffect" &&
     expr.statement()=="function_call")
  {
    // don't do function operand
    assert(expr.operands().size()==2);

    convert_expr(expr.op1()); // arguments
  }
  else
  {
    // fist do sub-nodes
    Forall_operands(it, expr)
      convert_expr(*it);
  }

  // now do case-split
  convert_expr_main(expr);
}

void llvm_adjust::convert_expr_main(exprt& expr)
{
  if(expr.id()=="sideeffect")
  {
    convert_side_effect(to_side_effect_expr(expr));
  }
  else if(expr.id()=="constant")
  {
  }
  else if(expr.id()=="symbol")
  {
    convert_symbol(expr);
  }
  else if(expr.id()=="unary+" || expr.id()=="unary-" ||
          expr.id()=="bitnot")
  {
  }
  else if(expr.id()=="not")
  {
    convert_expr_unary_boolean(expr);
  }
  else if(expr.is_and() || expr.is_or())
  {
    convert_expr_binary_boolean(expr);
  }
  else if(expr.is_address_of())
  {
  }
  else if(expr.is_dereference())
  {
    convert_dereference(expr);
  }
  else if(expr.is_member())
  {
    convert_member(to_member_expr(expr));
  }
  else if(expr.id()=="="  ||
          expr.id()=="notequal" ||
          expr.id()=="<"  ||
          expr.id()=="<=" ||
          expr.id()==">"  ||
          expr.id()==">=")
  {
  }
  else if(expr.is_index())
  {
    convert_index(to_index_expr(expr));
  }
  else if(expr.id()=="typecast")
  {
  }
  else if(expr.id() == "sizeof")
  {
    convert_sizeof(expr);
  }
  else if(expr.id()=="+" || expr.id()=="-" ||
          expr.id()=="*" || expr.id()=="/" ||
          expr.id()=="mod" ||
          expr.id()=="shl" || expr.id()=="shr" ||
          expr.id()=="bitand" || expr.id()=="bitxor" || expr.id()=="bitor")
  {
    convert_expr_binary_arithmetic(expr);
  }
  else if(expr.id()=="comma")
  {
  }
  else if(expr.id()=="if")
  {
    // If the condition is not of boolean type, it must be casted
    gen_typecast(ns, expr.op0(), bool_type());
  }
  else if(expr.is_code())
  {
    convert_code(to_code(expr));
  }
  else if(expr.id()=="builtin_offsetof")
  {
  }
  else if(expr.id()=="string-constant")
  {
  }
  else if(expr.id()=="arguments")
  {
  }
  else if(expr.id()=="union")
  {
  }
  else if(expr.id()=="struct")
  {
  }
  else
  {
    std::cout << "Unexpected expression: " << expr.id().as_string() << std::endl;
    expr.dump();
    abort();
  }
}

void llvm_adjust::convert_symbol(exprt& expr)
{
  const irep_idt &identifier=expr.identifier();

  // look it up
  symbolst::const_iterator s_it=context.symbols.find(identifier);

  if(s_it==context.symbols.end())
  {
    std::cout << "failed to find symbol `" << identifier << "'" << std::endl;
    abort();
  }

  // found it
  const symbolt &symbol=s_it->second;

  // save location
  locationt location=expr.location();

  if(symbol.is_macro)
  {
    expr=symbol.value;

    // put it back
    expr.location()=location;
  }
  else
  {
    expr=symbol_expr(symbol);

    // put it back
    expr.location()=location;

    if(symbol.lvalue)
      expr.cmt_lvalue(true);

    if(expr.type().is_code()) // function designator
    {
      // special case: this is sugar for &f
      address_of_exprt tmp(expr);
      tmp.implicit(true);
      tmp.location()=expr.location();
      expr.swap(tmp);
    }
  }
}

void llvm_adjust::convert_side_effect(side_effect_exprt& expr)
{
  const irep_idt &statement=expr.get_statement();

  if(statement=="preincrement" ||
     statement=="predecrement" ||
     statement=="postincrement" ||
     statement=="postdecrement")
  {
  }
  else if(has_prefix(id2string(statement), "assign"))
    convert_side_effect_assignment(expr);
  else if(statement=="function_call")
    convert_side_effect_function_call(to_side_effect_expr_function_call(expr));
  else if(statement=="statement_expression")
    convert_side_effect_statement_expression(expr);
  else
  {
    std::cout << "unknown side effect: " << statement;
    std::cout << "at " << expr.location() << std::endl;
    abort();
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

void llvm_adjust::convert_expr_binary_arithmetic(exprt& expr)
{
  exprt &op0=expr.op0();
  exprt &op1=expr.op1();

  const typet type0=ns.follow(op0.type());
  const typet type1=ns.follow(op1.type());

  if(expr.id()=="shl" || expr.id()=="shr")
  {
    gen_typecast_arithmetic(ns, op0);
    gen_typecast_arithmetic(ns, op1);

    if(is_number(op0.type()) &&
       is_number(op1.type()))
    {
      if(expr.id()=="shr") // shifting operation depends on types
      {
        if(type0.id()=="unsignedbv")
        {
          expr.id("lshr");
          return;
        }
        else if(type0.id()=="signedbv")
        {
          expr.id("ashr");
          return;
        }
      }

      return;
    }
  }
  else
  {
    gen_typecast_arithmetic(ns, op0, op1);

    const typet &type0=ns.follow(op0.type());
    const typet &type1=ns.follow(op1.type());

    if(expr.id()=="+" || expr.id()=="-" ||
       expr.id()=="*" || expr.id()=="/")
    {
      if(type0.id()=="pointer" || type1.id()=="pointer")
      {
//        typecheck_expr_pointer_arithmetic(expr);
        return;
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
        array_full_type.id()!="pointer" &&
        (index_full_type.is_array() ||
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
    deref.type()=op_type.subtype();
  }

  deref.cmt_lvalue(true);

  // if you dereference a pointer pointing to
  // a function, you get a pointer again
  // allowing ******...*p
  if(deref.type().is_code())
  {
    exprt tmp("address_of", pointer_typet());
    tmp.implicit(true);
    tmp.type().subtype()=deref.type();
    tmp.location()=deref.location();
    tmp.move_to_operands(deref);
    deref.swap(tmp);
  }
}

void llvm_adjust::convert_sizeof(exprt& expr)
{
  typet type;

  if(expr.operands().size()==0)
  {
    type = ((typet &)expr.sizeof_type());
    adjust_type(type);
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

void llvm_adjust::convert_side_effect_assignment(exprt& expr)
{
  const irep_idt &statement=expr.statement();

  exprt &op0=expr.op0();
  exprt &op1=expr.op1();

  const typet type0=op0.type();

  if(statement=="assign")
  {
    gen_typecast(ns, op1, type0);
    return;
  }
  else if(statement=="assign_shl" ||
          statement=="assign_shr")
  {
    gen_typecast_arithmetic(ns, op1);

    if(is_number(op1.type()))
    {
      if(statement=="assign_shl")
      {
        return;
      }
      else
      {
        if(type0.id()=="unsignedbv")
        {
          expr.statement("assign_lshr");
          return;
        }
        else if(type0.id()=="signedbv")
        {
          expr.statement("assign_ashr");
          return;
        }
      }
    }
  }
}

void llvm_adjust::convert_side_effect_function_call(
  side_effect_expr_function_callt& expr)
{
  exprt &f_op=expr.function();

  if(f_op.id()=="symbol")
  {
    if(context.symbols.find(f_op.identifier())==context.symbols.end())
    {
      // maybe this is an undeclared function
      // let's just add it
      const irep_idt &identifier=f_op.identifier();

      symbolt new_symbol;

      new_symbol.name=identifier;
      new_symbol.base_name=f_op.name();
      new_symbol.location=expr.location();
      new_symbol.type=f_op.type();
      new_symbol.type.incomplete(true);
      new_symbol.mode="C";

      symbolt *symbol_ptr;
      bool res = context.move(new_symbol, symbol_ptr);
      assert(!res);
      (void)res; // ndebug

      // LLVM will complain about this already, no need for us to do the same!
    }
  }

  convert_expr(f_op);

  // do implicit dereference
  if(f_op.is_address_of() &&
     f_op.implicit() &&
     f_op.operands().size()==1)
  {
    exprt tmp;
    tmp.swap(f_op.op0());
    f_op.swap(tmp);
  }
  else
  {
    exprt tmp("dereference", f_op.type().subtype());
    tmp.implicit(true);
    tmp.location()=f_op.location();
    tmp.move_to_operands(f_op);
    f_op.swap(tmp);
  }

  const code_typet &code_type = to_code_type(f_op.type());
  exprt::operandst &arguments = expr.arguments();
  const code_typet::argumentst &argument_types = code_type.arguments();

  for(unsigned i=0; i<arguments.size(); i++)
  {
    exprt &op=arguments[i];

    if(i<argument_types.size())
    {
      const code_typet::argumentt &argument_type=
        argument_types[i];

      const typet &op_type=argument_type.type();

      gen_typecast(ns, op, op_type);
    }
  }
}

void llvm_adjust::convert_side_effect_statement_expression(
  side_effect_exprt& expr)
{
  codet &code=to_code(expr.op0());
  assert(code.statement()=="block");

  // the type is the type of the last statement in the
  // block
  codet &last=to_code(code.operands().back());

  irep_idt last_statement=last.get_statement();

  if(last_statement=="expression")
  {
    assert(last.operands().size()==1);
    expr.type()=last.op0().type();
  }
  else if(last_statement=="function_call")
  {
    // make the last statement an expression

    code_function_callt &fc=to_code_function_call(last);

    side_effect_expr_function_callt sideeffect;

    sideeffect.function()=fc.function();
    sideeffect.arguments()=fc.arguments();
    sideeffect.location()=fc.location();

    sideeffect.type()=
      static_cast<const typet &>(fc.function().type().return_type());

    expr.type()=sideeffect.type();

    if(fc.lhs().is_nil())
    {
      codet code_expr("expression");
      code_expr.location() = fc.location();
      code_expr.move_to_operands(sideeffect);
      last.swap(code_expr);
    }
    else
    {
      codet code_expr("expression");
      code_expr.location() = fc.location();

      exprt assign("sideeffect");
      assign.statement("assign");
      assign.location()=fc.location();
      assign.move_to_operands(fc.lhs(), sideeffect);
      assign.type()=assign.op1().type();

      code_expr.move_to_operands(assign);
      last.swap(code_expr);
    }
  }
  else
    expr.type()=typet("empty");
}

void llvm_adjust::convert_expr_unary_boolean(exprt& expr)
{
  expr.type() = bool_type();

  exprt &operand=expr.op0();
  gen_typecast_bool(ns, operand);
}

void llvm_adjust::convert_expr_binary_boolean(exprt& expr)
{
  expr.type() = bool_type();

  gen_typecast_bool(ns, expr.op0());
  gen_typecast_bool(ns, expr.op1());
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
