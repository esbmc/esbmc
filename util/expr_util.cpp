/*******************************************************************\

Module:

Author: Daniel Kroening, kroening@kroening.com

\*******************************************************************/

#include "expr_util.h"
#include "fixedbv.h"
#include "ieee_float.h"
#include "bitvector.h"

/*******************************************************************\

Function: gen_zero

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

exprt gen_zero(const typet &type)
{
  exprt result;

  const std::string type_id=type.id_string();

  result=exprt("constant", type);

  if(type_id=="rational" ||
     type_id=="real" ||
     type_id=="integer" ||
     type_id=="natural" ||
     type_id=="complex")
  {
    result.set("value", "0");
  }
  else if(type_id=="unsignedbv" ||
          type_id=="signedbv" ||
          type_id=="verilogbv" ||
          type_id=="floatbv" ||
          type_id=="fixedbv" ||
          type_id=="c_enum")
  {
    std::string value;
    unsigned width=bv_width(type);

    for(unsigned i=0; i<width; i++)
      value+='0';

    result.set("value", value);
  }
  else if(type_id=="bool")
  {
    result.make_false();
  }
  else if(type_id=="pointer")
  {
    result.set("value", "NULL");
  }
  else
    result.make_nil();

  return result;
}

/*******************************************************************\

Function: gen_one

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

exprt gen_one(const typet &type)
{
  const std::string &type_id=type.id_string();
  exprt result=exprt("constant", type);

  if(type_id=="bool" ||
     type_id=="rational" ||
     type_id=="real" ||
     type_id=="integer" ||
     type_id=="natural" ||
     type_id=="complex")
  {
    result.set("value", "1");
  }
  else if(type_id=="unsignedbv" ||
          type_id=="signedbv")
  {
    std::string value;
    for(int i=0; i<atoi(type.get("width").c_str())-1; i++)
      value+='0';
    value+='1';
    result.set("value", value);
  }
  else if(type_id=="fixedbv")
  {
    fixedbvt fixedbv;
    fixedbv.spec=to_fixedbv_type(type);
    fixedbv.from_integer(1);
    result=fixedbv.to_expr();
  }
  else if(type_id=="floatbv")
  {
    ieee_floatt ieee_float;
    ieee_float.spec=to_floatbv_type(type);
    ieee_float.from_integer(1);
    result=ieee_float.to_expr();
  }
  else
    result.make_nil();

  return result;
}

/*******************************************************************\

Function: gen_not

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

exprt gen_not(const exprt &op)
{
  return gen_unary("not", typet("bool"), op);
}

/*******************************************************************\

Function: gen_unary

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

exprt gen_unary(const std::string &id, const typet &type, const exprt &op)
{
  exprt result(id, type);
  result.copy_to_operands(op);
  return result;
}

/*******************************************************************\

Function: gen_binary

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

exprt gen_binary(const std::string &id, const typet &type, const exprt &op1, const exprt &op2)
{
  exprt result(id, type);
  result.copy_to_operands(op1, op2);
  return result;
}

/*******************************************************************\

Function: gen_and

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

exprt gen_and(const exprt &op1, const exprt &op2)
{
  return gen_binary("and", typet("bool"), op1, op2);
}

/*******************************************************************\

Function: gen_and

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

exprt gen_and(const exprt &op1, const exprt &op2, const exprt &op3)
{
  exprt result("and", typet("bool"));
  result.copy_to_operands(op1, op2, op3);
  return result;
}

/*******************************************************************\

Function: gen_or

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

exprt gen_or(const exprt &op1, const exprt &op2)
{
  return gen_binary("or", typet("bool"), op1, op2);
}

/*******************************************************************\

Function: gen_or

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

exprt gen_or(const exprt &op1, const exprt &op2, const exprt &op3)
{
  exprt result("or", typet("bool"));
  result.copy_to_operands(op1, op2, op3);
  return result;
}

/*******************************************************************\

Function: gen_implies

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

exprt gen_implies(const exprt &op1, const exprt &op2)
{
  return gen_binary("=>", typet("bool"), op1, op2);
}

/*******************************************************************\

Function: gen_binary

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void gen_binary(exprt &expr, const std::string &id, bool default_value)
{
  if(expr.operands().size()==0)
  {
    if(default_value)
      expr.make_true();
    else
      expr.make_false();
  }
  else if(expr.operands().size()==1)
  {
    exprt tmp;
    tmp.swap(expr.op0());
    expr.swap(tmp);
  }
  else
  {
    expr.id(id);
    expr.type()=typet("bool");
  }
}

/*******************************************************************\

Function: gen_and

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void gen_and(exprt &expr)
{
  gen_binary(expr, "and", true);
}

/*******************************************************************\

Function: gen_or

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void gen_or(exprt &expr)
{
  gen_binary(expr, "or", false);
}

/*******************************************************************\

Function: symbol_expr

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

exprt symbol_expr(const symbolt &symbol)
{
  exprt tmp("symbol", symbol.type);
  tmp.set("identifier", symbol.name);
  return tmp;
}

/*******************************************************************\

Function: gen_pointer_type

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

pointer_typet gen_pointer_type(const typet &subtype)
{
  pointer_typet tmp;
  tmp.subtype()=subtype;
  return tmp;
}

/*******************************************************************\

Function: gen_address_of

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

exprt gen_address_of(const exprt &op)
{
  exprt tmp("address_of", gen_pointer_type(op.type()));
  tmp.copy_to_operands(op);
  return tmp;
}

/*******************************************************************\

Function: make_next_state

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

void make_next_state(exprt &expr)
{
  Forall_operands(it, expr)
    make_next_state(*it);
    
  if(expr.id()=="symbol")
    expr.id("next_symbol");
}

/*******************************************************************\

Function: make_binary

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

exprt make_binary(const exprt &expr)
{
  const exprt::operandst &operands=expr.operands();

  if(operands.size()<=2) return expr;

  exprt previous=operands[0];
  
  for(unsigned i=1; i<operands.size(); i++)
  {
    exprt tmp=expr;
    tmp.operands().clear();
    tmp.operands().resize(2);
    tmp.op0().swap(previous);
    tmp.op1()=operands[i];
    previous.swap(tmp);
  }
  
  return previous;
}

