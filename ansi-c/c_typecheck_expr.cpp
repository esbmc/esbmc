/*******************************************************************\

Module: C++ Language Type Checking

Author: Daniel Kroening, kroening@kroening.com

\*******************************************************************/

#include <assert.h>

#include <arith_tools.h>
#include <bitvector.h>
#include <config.h>
#include <expr_util.h>
#include <std_types.h>
#include <prefix.h>
#include <cprover_prefix.h>
#include <simplify_expr.h>
#include <base_type.h>
#include <std_expr.h>

#include "c_types.h"
#include "c_typecast.h"
#include "c_typecheck_base.h"
#include "c_sizeof.h"
#include "ansi_c_expr.h"

/*******************************************************************\

Function: c_typecheck_baset::typecheck_expr

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void c_typecheck_baset::typecheck_expr(exprt &expr)
{
  if(expr.id()=="already_typechecked")
  {
    assert(expr.operands().size()==1);
    exprt tmp;
    tmp.swap(expr.op0());
    expr.swap(tmp);
    return;
  }

  // fist do sub-nodes
  typecheck_expr_operands(expr);

  // now do case-split
  typecheck_expr_main(expr);
}

/*******************************************************************\

Function: c_typecheck_baset::typecheck_expr_main

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void c_typecheck_baset::typecheck_expr_main(exprt &expr)
{
  //std::cout << "expr.id(): " << expr.id() << std::endl;
  if(expr.id()=="sideeffect")
    typecheck_expr_side_effect(to_side_effect_expr(expr));
  else if(expr.id()=="constant")
    typecheck_expr_constant(expr);
  else if(expr.id()=="infinit")
  {
    // ignore
  }
  else if(expr.id()=="symbol")
    typecheck_expr_symbol(expr);
  else if(expr.id()=="unary+" || expr.id()=="unary-" ||
          expr.id()=="bitnot")
    typecheck_expr_unary_arithmetic(expr);
  else if(expr.id()=="not")
    typecheck_expr_unary_boolean(expr);
  else if(expr.is_and() || expr.id()=="or")
    typecheck_expr_binary_boolean(expr);
  else if(expr.is_address_of())
    typecheck_expr_address_of(expr);
  else if(expr.id()=="dereference")
    typecheck_expr_dereference(expr);
  else if(expr.id()=="member")
    typecheck_expr_member(expr);
  else if(expr.id()=="ptrmember")
    typecheck_expr_ptrmember(expr);
  else if(expr.id()=="="  ||
          expr.id()=="notequal" ||
          expr.id()=="<"  ||
          expr.id()=="<=" ||
          expr.id()==">"  ||
          expr.id()==">=")
    typecheck_expr_rel(expr);
  else if(expr.id()=="index")
    typecheck_expr_index(expr);
  else if(expr.id()=="typecast")
    typecheck_expr_typecast(expr);
  else if(expr.id()=="sizeof")
    typecheck_expr_sizeof(expr);
  else if(expr.id()=="+" || expr.id()=="-" ||
          expr.id()=="*" || expr.id()=="/" ||
          expr.id()=="mod" ||
          expr.id()=="shl" || expr.id()=="shr" ||
          expr.id()=="bitand" || expr.id()=="bitxor" || expr.id()=="bitor")
    typecheck_expr_binary_arithmetic(expr);
  else if(expr.id()=="comma")
    typecheck_expr_comma(expr);
  else if(expr.id()=="if")
    typecheck_expr_trinary(expr);
  else if(expr.is_code())
    typecheck_code(to_code(expr));
  else if(expr.id()=="builtin_va_arg")
    typecheck_expr_builtin_va_arg(expr);
  else if(expr.id()=="builtin_offsetof")
    typecheck_expr_builtin_offsetof(expr);
  else if(expr.id()=="string-constant")
  {
    // already fine
  }
  else if(expr.id()=="arguments")
  {
    // already fine
  }
  else if(expr.id()=="designated_initializer" ||
          expr.id()=="designated_list")
  {
    // already fine, just set type
    expr.type()=empty_typet();
  }
  else
  {
    err_location(expr);
    str << "unexpected expression: " << expr.pretty();
    throw 0;
  }
}

/*******************************************************************\

Function: c_typecheck_baset::typecheck_expr_comma

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void c_typecheck_baset::typecheck_expr_comma(exprt &expr)
{
  if(expr.operands().size()!=2)
  {
    err_location(expr);
    str << "operator `" << expr.id_string()
        << "' expects two operands";
    throw 0;
  }

  expr.type()=expr.op1().type();

  // make this an l-value if the last operand is one
  if(expr.op1().cmt_lvalue())
    expr.cmt_lvalue(true);
}

/*******************************************************************\

Function: c_typecheck_baset::typecheck_expr_builtin_va_arg

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void c_typecheck_baset::typecheck_expr_builtin_va_arg(exprt &expr)
{
  if(expr.operands().size()!=1)
  {
    err_location(expr);
    str << "builtin_va_arg expects one operand";
    throw 0;
  }

  typecheck_type(expr.type());
}

/*******************************************************************\

Function: c_typecheck_baset::typecheck_expr_builtin_offsetof

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void c_typecheck_baset::typecheck_expr_builtin_offsetof(exprt &expr)
{
  if(expr.operands().size()!=0)
  {
    err_location(expr);
    throw "builtin_offsetof expects no operands";
  }

  typet type=static_cast<const typet &>(expr.offsetof_type());
  typecheck_type(type);
  expr.offsetof_type(type);

  if(type.id()!="symbol")
  {
    err_location(expr);
    throw "builtin_offsetof expects struct type";
  }

  const irep_idt &identifier=type.identifier();
  symbolst::const_iterator s_it=context.symbols.find(identifier);

  if(s_it==context.symbols.end())
  {
    err_location(expr);
    str << "failed to find symbol `" << identifier << "'";
    throw 0;
  }

  // found it
  const symbolt &symbol=s_it->second;
  const irept &components=symbol.type.components();
  const irep_idt &member=expr.member_irep().identifier();
  bool found=false;
  mp_integer offset=0;

  forall_irep(it, components.get_sub())
  {
    if(it->name()==member)
    {
      found=true;
      break;
    }
    else
    {
      const typet &type=it->type();
      exprt size_expr=c_sizeof(type, *this);

      mp_integer i;
      to_integer(size_expr, i);
      offset+=i;
    }
  }

  if(!found)
  {
    err_location(expr);
    str << "builtin_offsetof references invalid member";
    throw 0;
  }

  exprt value_expr=from_integer(offset, uint_type());
  expr.swap(value_expr);
}

/*******************************************************************\

Function: c_typecheck_baset::typecheck_expr_operands

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void c_typecheck_baset::typecheck_expr_operands(exprt &expr)
{
  if(expr.id()=="sideeffect" &&
     expr.statement()=="function_call")
  {
    // don't do function operand
    assert(expr.operands().size()==2);

    typecheck_expr(expr.op1()); // arguments
  }
  else if(expr.id()=="sideeffect" &&
          expr.statement()=="statement_expression")
  {
    typecheck_code(to_code(expr.op0()));
  }
  else
  {
    Forall_operands(it, expr)
      typecheck_expr(*it);
  }
}

/*******************************************************************\

Function: c_typecheck_baset::typecheck_expr_symbol

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void c_typecheck_baset::typecheck_expr_symbol(exprt &expr)
{
  // adjust identifier, if needed
  replace_symbol(expr);

  const irep_idt &identifier=expr.identifier();

  // look it up
  symbolst::const_iterator s_it=context.symbols.find(identifier);

  if(s_it==context.symbols.end())
  {
    err_location(expr);
    str << "failed to find symbol `" << identifier << "'";
    throw 0;
  }

  // found it
  const symbolt &symbol=s_it->second;

  if(symbol.is_type)
  {
    err_location(expr);
    str << "did not expect a type symbol here, but got `"
        << symbol.display_name() << "'";
    throw 0;
  }

  // save location
  locationt location=expr.location();

  if(symbol.is_macro)
  {
    expr=symbol.value;

    // put it back
    expr.location()=location;
  }
  else if(has_prefix(id2string(identifier), CPROVER_PREFIX "constant_infinity"))
  {
    expr=exprt("infinity", symbol.type);

    // put it back
    expr.location()=location;
  }
  else if(identifier=="c::__func__")
  {
    // this is an ANSI-C standard compliant hack to get the function name
    string_constantt s;
    s.set_value(location.get_function());
    s.location()=location;
    expr.swap(s);
  }
  else
  {
    expr=symbol_expr(symbol);

    // put it back
    expr.location()=location;

    if(symbol.lvalue)
      expr.cmt_lvalue(true);

    if(expr.type().is_code()) // function designator
    { // special case: this is sugar for &f
      exprt tmp("address_of", pointer_typet());
      tmp.implicit(true);
      tmp.type().subtype()=expr.type();
      tmp.location()=expr.location();
      tmp.move_to_operands(expr);
      expr.swap(tmp);
    }
  }
}

/*******************************************************************\

Function: c_typecheck_baset::typecheck_side_effect_statement_expression

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void c_typecheck_baset::typecheck_side_effect_statement_expression(
  side_effect_exprt &expr)
{
  if(expr.operands().size()!=1)
  {
    err_location(expr);
    str << "statement expression expects one operand";
    throw 0;
  }

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

/*******************************************************************\

Function: c_typecheck_baset::typecheck_expr_sizeof

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void c_typecheck_baset::typecheck_expr_sizeof(exprt &expr)
{
  typet type;

  if(expr.operands().size()==0)
  {
    type = ((typet &)expr.sizeof_type());
    typecheck_type(type);
  }
  else if(expr.operands().size()==1)
  {
    type.swap(expr.op0().type());
  }
  else
  {
    err_location(expr);
    str << "sizeof operator expects zero or one operand, "
           "but got " << expr.operands().size();
    throw 0;
  }

  exprt new_expr=c_sizeof(type, *this);

  if(new_expr.is_nil())
  {
    err_location(expr);
    str << "type has no size: "
        << to_string(type);
    throw 0;
  }

  new_expr.swap(expr);

  expr.cmt_c_sizeof_type(type);
}

/*******************************************************************\

Function: c_typecheck_baset::typecheck_expr_typecast

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void c_typecheck_baset::typecheck_expr_typecast(exprt &expr)
{
  if(expr.operands().size()!=1)
  {
    err_location(expr);
    error("typecast operator expects one operand");
    throw 0;
  }

  exprt &op=expr.op0();

  typecheck_type(expr.type());

  const typet expr_type=follow(expr.type());
  const typet op_type=follow(op.type());

  // If we have a typecast from the same state to the same state, as generated
  // by CIL in places, this is just fine.
  if (full_eq(expr_type, op_type))
    return;

  if(expr_type.id()=="struct" ||
     expr_type.id()=="union")
  {
    // this is a GCC extension called 'temporary union'
    // the argument is expected to be a 'designated_list'
    if(op.id()!="designated_list")
    {
      err_location(expr);
      str << "type cast to struct or union requires designated_list "
             " argument";
      throw 0;
    }

    // build a constant from the argument
    // TODO: this may not yet work
    constant_exprt constant(expr_type);
    constant.operands().swap(op.operands());
    constant.location()=expr.location();
    expr.swap(constant);

    return;
  }

  if(!is_number(expr_type) &&
     !expr_type.is_bool() &&
     expr_type.id()!="pointer" &&
     !expr_type.is_array() &&
     expr_type.id()!="empty" &&
     expr_type.id()!="c_enum" &&
     expr_type.id()!="incomplete_c_enum")
  {
    err_location(expr);
    str << "type cast to `"
        << to_string(expr.type()) << "' not permitted";
    throw 0;
  }

  if(is_number(op_type) ||
     op_type.id()=="c_enum" ||
     op_type.id()=="incomplete_c_enum" ||
     op_type.is_bool() ||
     op_type.id()=="pointer")
  {
  }
  else if(op_type.is_array() ||
          op_type.id()=="incomplete_array")
  {
    index_exprt index;
    index.array()=op;
    index.index()=gen_zero(index_type());
    index.type()=op_type.subtype();
    op=gen_address_of(index);
  }
  else if(op_type.id()=="empty")
  {
    if(expr_type.id()!="empty")
    {
      err_location(expr);
      str << "type cast from void only permitted to void, but got `"
          << to_string(expr.type()) << "'";
      throw 0;
    }
  }
  else
  {
    err_location(expr);
    str << "type cast from `"
        << to_string(op_type) << "' not permitted";
    throw 0;
  }

  // special case: NULL

  if(expr_type.id()=="pointer" &&
     op.is_zero())
  {
    // zero typecasted to a pointer is NULL
    expr.id("constant");
    expr.remove("operands");
    expr.value("NULL");
    return;
  }

  // the new thing is an lvalue if the previous one is
  // an lvalue, and it's just a pointer type cast
  if(expr.op0().cmt_lvalue())
  {
    if(expr_type.id()=="pointer")
      expr.cmt_lvalue(true);
  }
}

/*******************************************************************\

Function: c_typecheck_baset::make_index_type

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void c_typecheck_baset::make_index_type(exprt &expr)
{
  const typet &full_type=follow(expr.type());

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
    err_location(expr);
    str << "expected integer type, but got `"
        << to_string(full_type) << "'";
    throw 0;
  }
}

/*******************************************************************\

Function: c_typecheck_baset::typecheck_expr_index

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void c_typecheck_baset::typecheck_expr_index(exprt &expr)
{
  if(expr.operands().size()!=2)
  {
    err_location(expr);
    str << "operator `" << expr.id_string()
        << "' expects two operands";
    throw 0;
  }

  exprt &array_expr=expr.op0();
  exprt &index_expr=expr.op1();

  // we might have to swap them

  {
    const typet &array_full_type=follow(array_expr.type());
    const typet &index_full_type=follow(index_expr.type());

    if(!array_full_type.is_array() &&
       array_full_type.id()!="incomplete_array" &&
       array_full_type.id()!="pointer" &&
       (index_full_type.is_array() ||
        index_full_type.id()=="incomplete_array" ||
        index_full_type.id()=="pointer"))
      std::swap(array_expr, index_expr);
  }

  make_index_type(index_expr);

  const typet &final_array_type=follow(array_expr.type());

  if(final_array_type.is_array() ||
     final_array_type.id()=="incomplete_array")
  {
    if(array_expr.cmt_lvalue())
      expr.cmt_lvalue(true);
  }
  else if(final_array_type.id()=="pointer")
  {
    // p[i] is syntactic sugar for *(p+i)

    exprt addition("+", array_expr.type());
    addition.operands().swap(expr.operands());
    expr.move_to_operands(addition);
    expr.id("dereference");
    expr.cmt_lvalue(true);
  }
  else
  {
    err_location(expr);
    str << "operator [] must take array or pointer but got `"
        << to_string(array_expr.type()) << "'";
    throw 0;
  }

  expr.type()=final_array_type.subtype();
}

/*******************************************************************\

Function: c_typecheck_baset::adjust_float_rel

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void c_typecheck_baset::adjust_float_rel(exprt &expr)
{
  // equality and disequality on float is not mathematical equality!
  assert(expr.operands().size()==2);

  if(follow(expr.op0().type()).id()=="floatbv")
  {
    if(expr.id()=="=")
      expr.id("ieee_float_equal");
    else if(expr.id()=="notequal")
      expr.id("ieee_float_notequal");
  }
}

/*******************************************************************\

Function: c_typecheck_baset::typecheck_expr_rel

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void c_typecheck_baset::typecheck_expr_rel(exprt &expr)
{
  expr.type()=typet("bool");

  if(expr.operands().size()!=2)
  {
    err_location(expr);
    str << "operator `" << expr.id_string()
        << "' expects two operands";
    throw 0;
  }

  exprt &op0=expr.op0();
  exprt &op1=expr.op1();

  const typet o_type0=op0.type();
  const typet o_type1=op1.type();

  if(expr.id()=="=" || expr.id()=="notequal")
  {
    if(follow(o_type0)==follow(o_type1))
    {
      const typet &final_type=follow(o_type0);
      if(!final_type.is_array() &&
         final_type.id()!="incomplete_array" &&
         final_type.id()!="incomplete_struct")
      {
        adjust_float_rel(expr);
        return; // no promotion necessary
      }
    }
  }

  implicit_typecast_arithmetic(op0, op1);

  const typet &type0=op0.type();
  const typet &type1=op1.type();

  if(type0==type1)
  {
    if(is_number(type0))
    {
      adjust_float_rel(expr);
      return;
    }

    if(type0.id()=="pointer")
    {
      if(expr.id()=="=" || expr.id()=="notequal")
        return;

      if(expr.id()=="<=" || expr.id()=="<" ||
         expr.id()==">=" || expr.id()==">")
        return;
    }

    if(type0.id()=="string-constant")
    {
      if(expr.id()=="=" || expr.id()=="notequal")
        return;
    }
  }
  else
  {
    // pointer and zero
    if(type0.id()=="pointer" && op1.is_zero())
    {
      op1=exprt("constant", type0);
      op1.value("NULL");
      return;
    }

    if(type1.id()=="pointer" && op0.is_zero())
    {
      op0=exprt("constant", type1);
      op0.value("NULL");
      return;
    }

    // pointer and integer
    if(type0.id()=="pointer" && is_number(type1))
    {
      op1.make_typecast(type0);
      return;
    }

    if(type1.id()=="pointer" && is_number(type0))
    {
      op0.make_typecast(type1);
      return;
    }

    if(type0.id()=="pointer" && type1.id()=="pointer")
    {
      op1.make_typecast(type0);
      return;
    }
  }

  err_location(expr);
  str << "operator `" << expr.id_string()
      << "' not defined for types `"
      << to_string(o_type0) << "' and `"
      << to_string(o_type1) << "'";
  throw 0;
}

/*******************************************************************\

Function: c_typecheck_baset::typecheck_expr_ptrmember

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void c_typecheck_baset::typecheck_expr_ptrmember(exprt &expr)
{
  if(expr.operands().size()!=1)
  {
    err_location(expr);
    error("ptrmember operator expects one operand");
    throw 0;
  }

  const typet &final_op0_type=follow(expr.op0().type());

  if(final_op0_type.id()!="pointer" &&
     !final_op0_type.is_array())
  {
    err_location(expr);
    str << "ptrmember operator requires pointer type "
           "on left hand side, but got `"
        << to_string(expr.op0().type()) << "'";
    throw 0;
  }

  // turn x->y into (*x).y

  exprt deref("dereference");
  deref.move_to_operands(expr.op0());
  deref.location()=expr.location();

  typecheck_expr_dereference(deref);

  expr.op0().swap(deref);

  expr.id("member");
  typecheck_expr_member(expr);
}

/*******************************************************************\

Function: c_typecheck_baset::typecheck_expr_member

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void c_typecheck_baset::typecheck_expr_member(exprt &expr)
{
  if(expr.operands().size()!=1)
  {
    err_location(expr);
    error("member operator expects one operand");
    throw 0;
  }

  exprt &op0=expr.op0();

  typet type=op0.type();

  follow_symbol(type);

  if(type.id()=="incomplete_struct")
  {
    err_location(expr);
    str << "member operator got incomplete structure type "
           "on left hand side";
    throw 0;
  }

  if(type.id()!="struct" &&
     type.id()!="union" &&
     type.id()!="class")
  {
    err_location(expr);
    str << "member operator requires structure type "
           "on left hand side but got `"
        << to_string(type) << "'";
    throw 0;
  }

  const irep_idt &component_name=
    expr.component_name();

  const irept &components=
    type.components();

  irept component;

  component.make_nil();

  forall_irep(it, components.get_sub())
    if(it->name()==component_name)
    {
      component=*it;
      break;
    }

  if(component.is_nil())
  {
    err_location(expr);
    str << "member `" << component_name
        << "' not found";
    throw 0;
  }

  const irep_idt &access=component.access();

  if(access=="private")
  {
    err_location(expr);
    str << "member `" << component_name
        << "' is " << access;
    throw 0;
  }

  expr.type()=component.type();

  if(op0.cmt_lvalue())
    expr.cmt_lvalue(true);

  if(op0.cmt_constant())
    expr.cmt_constant(true);

  // copy method identifier
  const irep_idt &identifier=component.cmt_identifier();

  if(identifier!="")
    expr.cmt_identifier(identifier);
}

/*******************************************************************\

Function: c_typecheck_baset::typecheck_expr_trinary

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void c_typecheck_baset::typecheck_expr_trinary(exprt &expr)
{
  exprt::operandst &operands=expr.operands();

  if(operands.size()!=3)
  {
    err_location(expr);
    error("Boolean operator ?: expects three operands");
    throw 0;
  }

  // copy (save) original types
  const typet o_type0=operands[0].type();
  const typet o_type1=operands[1].type();
  const typet o_type2=operands[2].type();

  implicit_typecast_bool(operands[0]);
  implicit_typecast_arithmetic(operands[1], operands[2]);

  if(operands[1].type().id()=="pointer" &&
     operands[2].type().id()!="pointer")
    implicit_typecast(operands[2], operands[1].type());
  else if(operands[2].type().id()=="pointer" &&
          operands[1].type().id()!="pointer")
    implicit_typecast(operands[1], operands[2].type());

  if(operands[1].type().id()=="pointer" &&
     operands[2].type().id()=="pointer" &&
     operands[1].type()!=operands[2].type())
  {
    // make it void *
    expr.type()=typet("pointer");
    expr.type().subtype()=typet("empty");
    implicit_typecast(operands[1], expr.type());
    implicit_typecast(operands[2], expr.type());
  }

  if(operands[1].type()==operands[2].type())
  {
    expr.type()=operands[1].type();
    return;
  }

  if(operands[1].type().id()=="empty" ||
     operands[2].type().id()=="empty")
  {
    expr.type()=empty_typet();
    return;
  }

  err_location(expr);
  str << "operator ?: not defined for types `"
      << to_string(o_type1) << "' and `"
      << to_string(o_type2) << "'";
  throw 0;
}

/*******************************************************************\

Function: c_typecheck_baset::typecheck_side_effect_gcc_conditional_expresssion

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void c_typecheck_baset::typecheck_side_effect_gcc_conditional_expression(
  side_effect_exprt &expr)
{
  exprt::operandst &operands=expr.operands();

  if(operands.size()!=2)
  {
    err_location(expr);
    error("gcc conditional_expr expects two operands");
    throw 0;
  }

  // use typechecking code for "if"

  exprt if_expr("if");
  if_expr.operands().resize(3);
  if_expr.op0()=operands[0];
  if_expr.op1()=operands[0];
  if_expr.op2()=operands[1];
  if_expr.location()=expr.location();

  typecheck_expr_trinary(if_expr);

  // copy the result
  expr.op0()=if_expr.op1();
  expr.op1()=if_expr.op2();
  expr.type()=if_expr.type();
}

/*******************************************************************\

Function: c_typecheck_baset::typecheck_expr_address_of

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void c_typecheck_baset::typecheck_expr_address_of(exprt &expr)
{
  if(expr.operands().size()!=1)
  {
    err_location(expr);
    error("unary operator & expects one operand");
    throw 0;
  }

  exprt &op=expr.op0();

  // special case: address of function designator
  // ANSI-C 99 section 6.3.2.1 paragraph 4

  if(op.is_address_of() &&
     op.implicit() &&
     op.operands().size()==1 &&
     op.op0().id()=="symbol" &&
     op.op0().type().is_code())
  {
    // make the implicit address_of an explicit address_of
    exprt tmp;
    tmp.swap(op);
    tmp.implicit(false);
    expr.swap(tmp);
    return;
  }

  expr.type()=typet("pointer");

  if(!op.type().is_code() &&
     !op.cmt_lvalue())
  {
    err_location(expr);
    str << "address_of error: `" << to_string(op)
        << "' not an lvalue";
    throw 0;
  }

  // turn &array into &(array[0])
  if(follow(op.type()).is_array())
  {
    index_exprt index;
    index.array()=op;
    index.index()=gen_zero(index_type());
    index.type()=follow(op.type()).subtype();
    index.location()=expr.location();
    op.swap(index);
  }

  expr.type().subtype()=op.type();
}

/*******************************************************************\

Function: c_typecheck_baset::typecheck_expr_dereference

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void c_typecheck_baset::typecheck_expr_dereference(exprt &expr)
{
  if(expr.operands().size()!=1)
  {
    err_location(expr);
    str << "unary operator * expects one operand";
    throw 0;
  }

  exprt &op=expr.op0();

  const typet op_type=follow(op.type());

  if(op_type.is_array() ||
     op_type.id()=="incomplete_array")
  {
    // *a is the same as a[0]
    expr.id("index");
    expr.type()=op_type.subtype();
    expr.copy_to_operands(gen_zero(index_type()));
    assert(expr.operands().size()==2);
  }
  else if(op_type.id()=="pointer")
  {
    if(op_type.subtype().id()=="empty")
    {
      err_location(expr);
      error("operand of unary * is a void * pointer");
      throw 0;
    }

    expr.type()=op_type.subtype();
  }
  else
  {
    err_location(expr);
    str << "operand of unary * `" << to_string(op)
        << "' is not a pointer";
    throw 0;
  }

  expr.cmt_lvalue(true);

  // if you dereference a pointer pointing to
  // a function, you get a pointer again
  // allowing ******...*p

  typecheck_expr_function_identifier(expr);
}

/*******************************************************************\

Function: c_typecheck_baset::typecheck_expr_function_identifier

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void c_typecheck_baset::typecheck_expr_function_identifier(exprt &expr)
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

/*******************************************************************\

Function: c_typecheck_baset::typecheck_expr_side_effect

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void c_typecheck_baset::typecheck_expr_side_effect(side_effect_exprt &expr)
{
  const irep_idt &statement=expr.get_statement();

  if(statement=="preincrement" ||
     statement=="predecrement" ||
     statement=="postincrement" ||
     statement=="postdecrement")
  {
    if(expr.operands().size()!=1)
    {
      err_location(expr);
      str << statement << "operator expects one operand";
    }

    const exprt &op0=expr.op0();
    const typet &type0=op0.type();
    const typet &final_type0=follow(type0);

    if(!op0.cmt_lvalue())
    {
      err_location(op0);
      str << "prefix operator error: `" << to_string(op0)
          << "' not an lvalue";
      throw 0;
    }

    if(type0.cmt_constant())
    {
      err_location(op0);
      std::string msg =  "warning: `" + to_string(op0) + "' is constant";
      warning(msg);
    }

    if(is_number(final_type0) ||
       final_type0.is_bool() ||
       final_type0.id()=="c_enum" ||
       final_type0.id()=="incomplete_c_enum" ||
       final_type0.id()=="pointer")
    {
      expr.type()=type0;
    }
    else
    {
      err_location(expr);
      str << "operator `" << statement
          << "' not defined for type `"
          << to_string(type0) << "'";
      throw 0;
    }
  }
  else if(has_prefix(id2string(statement), "assign"))
    typecheck_side_effect_assignment(expr);
  else if(statement=="function_call")
    typecheck_side_effect_function_call(to_side_effect_expr_function_call(expr));
  else if(statement=="statement_expression")
    typecheck_side_effect_statement_expression(expr);
  else if(statement=="gcc_conditional_expression")
    typecheck_side_effect_gcc_conditional_expression(expr);
  else
  {
    err_location(expr);
    str << "unknown side effect: " << statement;
    throw 0;
  }
}

/*******************************************************************\

Function: c_typecheck_baset::typecheck_side_effect_function_call

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void c_typecheck_baset::typecheck_side_effect_function_call(
  side_effect_expr_function_callt &expr)
{
  if(expr.operands().size()!=2)
  {
    err_location(expr);
    throw "function_call side effect expects two operands";
  }

  exprt &f_op=expr.function();

  // f_op is not yet typechecked, in contrast to the other arguments
  // this is a big special case

  if(f_op.id()=="symbol")
  {
    replace_symbol(f_op);

    if(context.symbols.find(f_op.identifier())==context.symbols.end())
    {
      // maybe this is an undeclared function
      // let's just add it
      const irep_idt &identifier=f_op.identifier();

      symbolt new_symbol;

      new_symbol.name=identifier;
      new_symbol.base_name=std::string(id2string(identifier), 3, std::string::npos);
      new_symbol.location=expr.location();
      new_symbol.type=code_typet();
      new_symbol.type.incomplete(true);
      new_symbol.type.return_type(int_type());
      // TODO: should add arguments

      symbolt *symbol_ptr;
      bool res = move_symbol(new_symbol, symbol_ptr);
      assert(!res);

      err_location(f_op);
      str << "function `" << identifier << "' is not declared";
      warning();
    }
  }

  // typecheck it now
  typecheck_expr(f_op);

  const typet f_op_type=follow(f_op.type());

  if(f_op_type.id()!="pointer")
  {
    err_location(f_op);
    str << "expected function/function pointer as argument but got `"
        << to_string(f_op_type) << "'";
    throw 0;
  }

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
    exprt tmp("dereference", f_op_type.subtype());
    tmp.implicit(true);
    tmp.location()=f_op.location();
    tmp.move_to_operands(f_op);
    f_op.swap(tmp);
  }

  if(!f_op.type().is_code())
  {
    err_location(f_op);
    throw "expected code as argument";
  }

  const code_typet &code_type=to_code_type(f_op.type());

  expr.type()=code_type.return_type();

  typecheck_function_call_arguments(expr);

  do_special_functions(expr);
}

/*******************************************************************\

Function: c_typecheck_baset::do_special_functions

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void c_typecheck_baset::do_special_functions(
  side_effect_expr_function_callt &expr)
{
  const exprt &f_op=expr.function();

  // some built-in functions
  if(f_op.id()=="symbol")
  {
    const irep_idt &identifier=to_symbol_expr(f_op).get_identifier();

    if(identifier==CPROVER_PREFIX "same_object")
    {
      if(expr.arguments().size()!=2)
      {
        err_location(f_op);
        throw "same_object expects two operands";
      }

      exprt same_object_expr("same-object", bool_typet());
      same_object_expr.operands()=expr.arguments();
      expr.swap(same_object_expr);
      //std::cout << "expr.pretty(): " << expr.pretty() << std::endl;
    }
    else if(identifier==CPROVER_PREFIX "buffer_size")
    {
      if(expr.arguments().size()!=1)
      {
        err_location(f_op);
        throw "buffer_size expects one operand";
      }

      exprt buffer_size_expr("buffer_size", uint_type());
      buffer_size_expr.operands()=expr.arguments();
      expr.swap(buffer_size_expr);
    }
    else if(identifier==CPROVER_PREFIX "is_zero_string")
    {
      if(expr.arguments().size()!=1)
      {
        err_location(f_op);
        throw "is_zero_string expects one operand";
      }

      exprt is_zero_string_expr("is_zero_string", bool_typet());
      is_zero_string_expr.operands()=expr.arguments();
      is_zero_string_expr.cmt_lvalue(true); // make it an lvalue
      expr.swap(is_zero_string_expr);
    }
    else if(identifier==CPROVER_PREFIX "zero_string_length")
    {
      if(expr.arguments().size()!=1)
      {
        err_location(f_op);
        throw "zero_string_length expects one operand";
      }

      exprt zero_string_length_expr("zero_string_length", uint_type());
      zero_string_length_expr.operands()=expr.arguments();
      zero_string_length_expr.cmt_lvalue(true); // make it an lvalue
      expr.swap(zero_string_length_expr);
    }
    else if(identifier==CPROVER_PREFIX "POINTER_OFFSET")
    {
      if(expr.arguments().size()!=1)
        throw "pointer_offset expects one argument";

      exprt pointer_offset_expr=exprt("pointer_offset", expr.type());
      pointer_offset_expr.operands()=expr.arguments();
      expr.swap(pointer_offset_expr);
    }
    else if(identifier==CPROVER_PREFIX "POINTER_OBJECT")
    {
      if(expr.arguments().size()!=1)
        throw "pointer_object expects one argument";

      exprt pointer_object_expr=exprt("pointer_object", expr.type());
      pointer_object_expr.operands()=expr.arguments();
      expr.swap(pointer_object_expr);
    }
    else if(identifier==CPROVER_PREFIX "isnan")
    {
      if(expr.arguments().size()!=1)
      {
        err_location(f_op);
        throw "isnan expects one operand";
      }

      exprt isnan_expr("isnan", bool_typet());
      isnan_expr.operands()=expr.arguments();
      expr.swap(isnan_expr);
    }
    else if(identifier==CPROVER_PREFIX "isfinite")
    {
      if(expr.arguments().size()!=1)
      {
        err_location(f_op);
        throw "isfinite expects one operand";
      }

      exprt isfinite_expr("isfinite", bool_typet());
      isfinite_expr.operands()=expr.arguments();
      expr.swap(isfinite_expr);
    }
    else if(identifier==CPROVER_PREFIX "abs" ||
            identifier==CPROVER_PREFIX "fabs" ||
            identifier==CPROVER_PREFIX "fabsf" ||
            identifier==CPROVER_PREFIX "fabsl")
    {
      if(expr.arguments().size()!=1)
      {
        err_location(f_op);
        throw "abs expects one operand";
      }

      exprt abs_expr("abs", expr.type());
      abs_expr.operands()=expr.arguments();
      expr.swap(abs_expr);
    }
    else if(identifier==CPROVER_PREFIX "isinf")
    {
      if(expr.arguments().size()!=1)
      {
        err_location(f_op);
        throw "isinf expects one operand";
      }

      exprt isinf_expr("isinf", bool_typet());
      isinf_expr.operands()=expr.arguments();
      expr.swap(isinf_expr);
    }
    else if(identifier==CPROVER_PREFIX "isnormal")
    {
      if(expr.arguments().size()!=1)
      {
        err_location(f_op);
        throw "finite expects one operand";
      }

      exprt isnormal_expr("isnormal", bool_typet());
      isnormal_expr.operands()=expr.arguments();
      expr.swap(isnormal_expr);
    }
    else if(identifier==CPROVER_PREFIX "sign")
    {
      if(expr.arguments().size()!=1)
      {
        err_location(f_op);
        throw "sign expects one operand";
      }

      exprt sign_expr("sign", bool_typet());
      sign_expr.operands()=expr.arguments();
      expr.swap(sign_expr);
    }
    else if(identifier=="c::__builtin_expect")
    {
      // this is a gcc extension to provide branch prediction
      if(expr.arguments().size()!=2)
      {
        err_location(f_op);
        throw "__builtin_expect expects two arguments";
      }

      exprt tmp=expr.arguments()[0];
      expr.swap(tmp);
    }
  }
}

/*******************************************************************\

Function: c_typecheck_baset::typecheck_function_call_arguments

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void c_typecheck_baset::typecheck_function_call_arguments(
  side_effect_expr_function_callt &expr)
{
  exprt &f_op=expr.function();
  const code_typet &code_type=to_code_type(f_op.type());
  exprt::operandst &arguments=expr.arguments();
  const code_typet::argumentst &argument_types=
    code_type.arguments();

  // no. of arguments test

  if(code_type.incomplete())
  {
    // can't check
  }
  else if(code_type.has_ellipsis())
  {
    if(argument_types.size()>arguments.size())
    {
      err_location(expr);
      throw "not enough arguments";
    }
  }
  else if(argument_types.size()!=arguments.size())
  {
    err_location(expr);
    str << "wrong number of arguments: "
        << "expected " << argument_types.size()
        << ", but got " << arguments.size();
    throw 0;
  }

  for(unsigned i=0; i<arguments.size(); i++)
  {
    exprt &op=arguments[i];

    if(i<argument_types.size())
    {
      const code_typet::argumentt &argument_type=
        argument_types[i];

      const typet &op_type=argument_type.type();

      if(op_type.is_bool() &&
         op.id()=="sideeffect" &&
         op.statement()=="assign" &&
         !op.type().is_bool())
      {
        err_location(expr);
        warning("assignment where Boolean argument is expected");
      }

      implicit_typecast(op, op_type);
    }
    else
    {
      // don't know type, just do standard conversion

      const typet &type=follow(op.type());
      if(type.is_array() || type.id()=="incomplete_array")
        implicit_typecast(op, pointer_typet(empty_typet()));
    }
  }
}

/*******************************************************************\

Function: c_typecheck_baset::typecheck_expr_constant

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void c_typecheck_baset::typecheck_expr_constant(exprt &expr)
{
  const typet &type=expr.type();

  if(type.id()=="integer")
  {
    // save location
    locationt location=expr.location();

    mp_integer value=string2integer(expr.value().as_string());
    const std::string &given_width=expr.type().cmt_width().as_string();
    bool is_unsigned=expr.type().cmt_unsigned();
    bool is_hex_or_oct=expr.hex_or_oct();
    const std::string cformat=expr.cformat().as_string();

    if(value<0)
      throw "unexpected value";

    unsigned min_width=config.ansi_c.int_width;

    if(given_width=="long")
      min_width=config.ansi_c.long_int_width;
    else if(given_width=="longlong")
      min_width=config.ansi_c.long_long_int_width;
    else if(given_width=="8" || given_width=="16" ||
            given_width=="32" || given_width=="64" ||
            given_width=="128")
      min_width=atoi(given_width.c_str());
    else if(given_width!="")
    {
      err_location(expr);
      str << "unknown width: " << given_width;
      throw 0;
    }

    #define FITS(width, signed) \
      ((signed?!is_unsigned:(is_unsigned || is_hex_or_oct)) && \
      (width>=min_width) && \
      (power(2, signed?width-1:width)>value))

    unsigned width;
    bool is_signed=false;

    if(FITS(config.ansi_c.int_width, true)) // int
    {
      width=config.ansi_c.int_width;
      is_signed=true;
    }
    else if(FITS(config.ansi_c.int_width, false)) // unsigned int
    {
      width=config.ansi_c.int_width;
    }
    else if(FITS(config.ansi_c.long_int_width, true)) // long int
    {
      width=config.ansi_c.long_int_width;
      is_signed=true;
    }
    else if(FITS(config.ansi_c.long_int_width, false)) // unsigned long int
    {
      width=config.ansi_c.long_int_width;
    }
    else if(FITS(config.ansi_c.long_long_int_width, true)) // long long int
    {
      width=config.ansi_c.long_long_int_width;
      is_signed=true;
    }
    else if(FITS(config.ansi_c.long_long_int_width, false)) // unsigned long long int
    {
      width=config.ansi_c.long_long_int_width;
    }
    else
    {
      err_location(expr);
      str << "constant " << value << " does not fit any scalar type";
      throw 0;
    }

    typet value_type(is_signed?"signedbv":"unsignedbv");
    value_type.width(width);

    expr=from_integer(value, value_type);
    expr.location()=location;

    if(!cformat.empty())
      expr.cformat(cformat);
  }
  else if(type.is_array() ||
          type.id()=="incomplete_array")
  {
    // nothing to do
  }
}

/*******************************************************************\

Function: c_typecheck_baset::typecheck_expr_unary_arithmetic

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void c_typecheck_baset::typecheck_expr_unary_arithmetic(exprt &expr)
{
  if(expr.operands().size()!=1)
  {
    err_location(expr);
    str << "operator `" << expr.id_string()
        << "' expects one operand";
    throw 0;
  }

  exprt &operand=expr.op0();

  implicit_typecast_arithmetic(operand);

  if(is_number(operand.type()))
  {
    expr.type()=operand.type();
    return;
  }

  err_location(expr);
  str << "operator `" << expr.id_string()
      << "' not defined for type `"
      << to_string(operand.type()) << "'";
  throw 0;
}

/*******************************************************************\

Function: c_typecheck_baset::typecheck_expr_unary_boolean

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void c_typecheck_baset::typecheck_expr_unary_boolean(exprt &expr)
{
  if(expr.operands().size()!=1)
  {
    err_location(expr);
    str << "operator `" << expr.id_string()
        << "' expects one operand";
    throw 0;
  }

  exprt &operand=expr.op0();

  implicit_typecast_bool(operand);
  expr.type()=typet("bool");
}

/*******************************************************************\

Function: c_typecheck_baset::typecheck_expr_binary_arithmetic

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void c_typecheck_baset::typecheck_expr_binary_arithmetic(exprt &expr)
{
  if(expr.operands().size()!=2)
  {
    err_location(expr);
    str << "operator `" << expr.id_string()
        << "' expects two operands";
    throw 0;
  }

  exprt &op0=expr.op0();
  exprt &op1=expr.op1();

  const typet o_type0=follow(op0.type());
  const typet o_type1=follow(op1.type());

  if(expr.id()=="shl" || expr.id()=="shr")
  {
    // do them separately!
    implicit_typecast_arithmetic(op0);
    implicit_typecast_arithmetic(op1);

    if(is_number(op0.type()) &&
       is_number(op1.type()))
    {
      expr.type()=op0.type();

      if(expr.id()=="shr") // shifting operation depends on types
      {
        const typet &op0_type=follow(op0.type());

        if(op0_type.id()=="unsignedbv")
        {
          expr.id("lshr");
          return;
        }
        else if(op0_type.id()=="signedbv")
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
    implicit_typecast_arithmetic(op0, op1);

    const typet &type0=follow(op0.type());
    const typet &type1=follow(op1.type());

    if(expr.id()=="+" || expr.id()=="-" ||
       expr.id()=="*" || expr.id()=="/")
    {
      if(type0.id()=="pointer" || type1.id()=="pointer")
      {
        typecheck_expr_pointer_arithmetic(expr);
        return;
      }
      else if(type0==type1)
      {
        if(is_number(type0))
        {
          expr.type()=type0;
          return;
        }
      }
    }
    else if(expr.id()=="mod")
    {
      if(type0==type1)
      {
        if(type0.id()=="signedbv" || type0.id()=="unsignedbv")
        {
          expr.type()=type0;
          return;
        }
      }
    }
    else if(expr.id()=="bitand" || expr.id()=="bitxor" || expr.id()=="bitor")
    {
      if(type0==type1)
      {
        if(is_number(type0))
        {
          expr.type()=type0;
          return;
        }
      }
    }
  }

  err_location(expr);
  str << "operator `" << expr.id_string()
      << "' not defined for types `"
      << to_string(o_type0) << "' and `"
      << to_string(o_type1) << "'";
  throw 0;
}

/*******************************************************************\

Function: c_typecheck_baset::typecheck_expr_pointer_arithmetic

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void c_typecheck_baset::typecheck_expr_pointer_arithmetic(exprt &expr)
{
  exprt &op0=expr.op0();
  exprt &op1=expr.op1();

  const typet &type0=op0.type();
  const typet &type1=op1.type();

  if(expr.id()=="-" ||
     (expr.id()=="sideeffect" && expr.statement()=="assign-"))
  {
    if(type0.id()=="pointer" &&
       type1.id()=="pointer")
    {
      typet pointer_diff_type;

      pointer_diff_type=typet("signedbv");
      pointer_diff_type.width(config.ansi_c.pointer_diff_width);

      expr.type()=pointer_diff_type;
      return;
    }

    if(type0.id()=="pointer")
    {
      make_index_type(op1);
      expr.type()=type0;
      return;
    }
  }
  else if(expr.id()=="+" ||
          (expr.id()=="sideeffect" && expr.statement()=="assign+"))
  {
    exprt *pop, *intop;

    if(type0.id()=="pointer")
    {
      pop=&op0;
      intop=&op1;
    }
    else if(type1.id()=="pointer")
    {
      pop=&op1;
      intop=&op0;
    }
    else
      assert(false);

    make_index_type(*intop);
    expr.type()=pop->type();
    return;
  }

  err_location(expr);
  str << "operator `" << expr.id_string()
      << "' not defined for types `"
      << to_string(type0) << "' and `"
      << to_string(type1) << "'";
  throw 0;
}

/*******************************************************************\

Function: c_typecheck_baset::typecheck_expr_binary_boolean

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void c_typecheck_baset::typecheck_expr_binary_boolean(exprt &expr)
{
  if(expr.operands().size()!=2)
  {
    err_location(expr);
    str << "operator `" << expr.id_string()
        << "' expects two operands";
    throw 0;
  }

  implicit_typecast_bool(expr.op0());
  implicit_typecast_bool(expr.op1());

  expr.type()=typet("bool");
}

/*******************************************************************\

Function: c_typecheck_baset::typecheck_side_effect_assignment

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void c_typecheck_baset::typecheck_side_effect_assignment(exprt &expr)
{
  if(expr.operands().size()!=2)
  {
    err_location(expr);
    str << "operator `" << expr.statement()
        << "' expects two operands";
    throw 0;
  }

  const irep_idt &statement=expr.statement();

  exprt &op0=expr.op0();
  exprt &op1=expr.op1();

  // se if we have a typecast on the LHS
  if(op0.id()=="typecast")
  {
    assert(op0.operands().size()==1);

    // set #lvalue and #constant
    op0.cmt_lvalue(op0.op0().cmt_lvalue());
    op0.cmt_constant(op0.op0().cmt_constant());
  }

  const typet o_type0=op0.type();
  const typet o_type1=op1.type();

  const typet &type0=op0.type();
  const typet &final_type0=follow(type0);
  //const typet &type1=op1.type();

  expr.type()=type0;

  if(!op0.cmt_lvalue())
  {
    err_location(expr);
    str << "assignment error: `" << to_string(op0)
        << "' not an lvalue";
    throw 0;
  }

  if(o_type0.cmt_constant())
  {
    err_location(expr);
    std::string msg = "warning: `" + to_string(op0) + "' is constant";
    warning(msg);
  }

  if(statement=="assign")
  {
    implicit_typecast(op1, o_type0);
    return;
  }
  else if(statement=="assign_shl" ||
          statement=="assign_shr")
  {
    implicit_typecast_arithmetic(op1);

    if(is_number(op1.type()))
    {
      expr.type()=type0;

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
  else
  {
    if(final_type0.id()=="pointer" &&
       (statement=="assign-" || statement=="assign+"))
    {
      typecheck_expr_pointer_arithmetic(expr);
      return;
    }
    else if(final_type0.is_bool() ||
            final_type0.id()=="c_enum" ||
            final_type0.id()=="incomplete_c_enum")
    {
      implicit_typecast_arithmetic(op1);
      if(is_number(op1.type()))
        return;
    }
    else
    {
      implicit_typecast(op1, op0.type());
      if(is_number(op0.type()))
      {
        expr.type()=type0;
        return;
      }
    }
  }

  err_location(expr);
  str << "assignment `" << statement
      << "' not defined for types `"
      << to_string(o_type0) << "' and `"
      << to_string(o_type1) << "'";

  throw 0;
}

/*******************************************************************\

Function: c_typecheck_baset::make_constant

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void c_typecheck_baset::make_constant(exprt &expr)
{
  make_constant_rec(expr);
  base_type(expr, *this);
  simplify(expr);

  if(!expr.is_constant() &&
     expr.id()!="infinity")
  {
    err_location(expr.find_location());
    str << "expected constant expression, but got `"
        << to_string(expr) << "'";
    throw 0;
  }
}

/*******************************************************************\

Function: c_typecheck_baset::make_constant_index

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void c_typecheck_baset::make_constant_index(exprt &expr)
{
  make_constant(expr);

  if(expr.id()!="infinity")
  {
    make_index_type(expr);
    simplify(expr);

    if(!expr.is_constant())
    {
      err_location(expr.find_location());
      throw "conversion to integer failed";
    }
  }
}

/*******************************************************************\

Function: c_typecheck_baset::make_constant_rec

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void c_typecheck_baset::make_constant_rec(exprt &expr __attribute__((unused)))
{
}
