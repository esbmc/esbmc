/*******************************************************************\

Module:

Author: Daniel Kroening, kroening@kroening.com

\*******************************************************************/

#include <string.h>

#include <lispexpr.h>
#include <lispirep.h>
#include <bitvector.h>
#include <i2string.h>
#include <std_code.h>

#include "expr2bp.h"

class expr2bpt
{
public:
  std::string convert_nondet_bool(const exprt &src, unsigned precedence);

  std::string convert_binary(const exprt &src, const std::string &symbol,
                             unsigned precedence);

  std::string convert_implies(
    const exprt &src,
    unsigned precedence);

  std::string convert_unary(const exprt &src, const std::string &symbol,
                            unsigned precedence);

  std::string convert_index(const exprt &src, unsigned precedence);

  std::string convert(const exprt &src, unsigned &precedence);

  std::string convert(const exprt &src);

  std::string convert_symbol(const exprt &src, unsigned &precedence);

  std::string convert_bool_vector(const exprt &src, unsigned &precedence);

  std::string convert_predicate_symbol(const exprt &src, unsigned &precedence);

  std::string convert_next_symbol(const exprt &src, unsigned &precedence);

  std::string convert_constant(const exprt &src, unsigned &precedence);

  std::string convert_sideeffect(const exprt &src, unsigned &precedence);

  std::string convert_code(const codet &src);

  std::string convert_code_goto(const codet &src);

  std::string convert_code_return(const code_returnt &src);

  std::string convert_code_assign(const code_assignt &src);

  std::string convert_code_bp_constrain(const codet &src);

  std::string convert_code_decl(const code_declt &src);

  std::string convert_code_ifthenelse(const codet &src);

  std::string convert_code_function_call(const code_function_callt &src);

  std::string convert_norep(const exprt &src, unsigned &precedence);

  std::string convert(const typet &src);
};

/*******************************************************************\

Function: expr2bpt::convert_nondet_bool

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

std::string expr2bpt::convert_nondet_bool(
  const exprt &src,
  unsigned precedence)
{
  return "*";
}

/*******************************************************************\

Function: expr2bpt::convert_binary

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

std::string expr2bpt::convert_binary(
  const exprt &src,
  const std::string &symbol,
  unsigned precedence)
{
  if(src.operands().size()<2)
    return convert_norep(src, precedence);

  bool first=true;
  std::string dest;

  forall_operands(it, src)
  {
    if(first)
      first=false;
    else
    {
      dest+=' ';
      dest+=symbol;
      dest+=' ';
    }

    unsigned p;
    std::string op=convert(*it, p);

    if(precedence>p) dest+='(';
    dest+=op;
    if(precedence>p) dest+=')';
  }

  return dest;
}

/*******************************************************************\

Function: expr2bpt::convert_implies

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

std::string expr2bpt::convert_implies(
  const exprt &src,
  unsigned precedence)
{
  if(src.operands().size()!=2)
    return convert_norep(src, precedence);

  std::string dest;
  
  dest="!("+convert(src.op0())+") | ";

  unsigned p;  
  std::string op1=convert(src.op1(), p);
  
  if(precedence>p) dest+='(';
  dest+=op1;
  if(precedence>p) dest+=')';

  return dest;
}

/*******************************************************************\

Function: expr2bpt::convert_unary

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

std::string expr2bpt::convert_unary(
  const exprt &src,
  const std::string &symbol,
  unsigned precedence)
{
  if(src.operands().size()!=1)
    return convert_norep(src, precedence);
    
  unsigned p;
  std::string op=convert(src.op0(), p);

  std::string dest=symbol;

  if(precedence>p) dest+='(';
  dest+=op;
  if(precedence>p) dest+=')';

  return dest;
}

/*******************************************************************\

Function: expr2bpt::convert_index

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

std::string expr2bpt::convert_index(
  const exprt &src,
  unsigned precedence)
{
  if(src.operands().size()!=2)
    return convert_norep(src, precedence);

  unsigned p;
  std::string op0=convert(src.op0(), p);

  std::string dest;

  if(precedence>p) dest+='(';
  dest+=op0;
  if(precedence>p) dest+=')';

  std::string op1=convert(src.op1(), p);

  dest+='[';
  dest+=op1;
  dest+=']';

  return dest;
}

/*******************************************************************\

Function: expr2bpt::convert_norep

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

std::string expr2bpt::convert_norep(
  const exprt &src,
  unsigned &precedence)
{
  precedence=22;
  return src.to_string();
}

/*******************************************************************\

Function: expr2bpt::convert_symbol

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

std::string expr2bpt::convert_symbol(
  const exprt &src,
  unsigned &precedence)
{
  precedence=22;
  std::string dest=src.get_string("identifier");
 
  if(strncmp(dest.c_str(), "bp::", 4)==0)
  {
    dest.erase(0, 4);

    if(strncmp(dest.c_str(), "var::", 5)==0)
      dest.erase(0, 5);
    else if(strncmp(dest.c_str(), "local_var::", 11)==0)
      dest.erase(0, 11);
  }

  return dest;
}

/*******************************************************************\

Function: expr2bpt::convert_bool_vector

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

std::string expr2bpt::convert_bool_vector(
  const exprt &src,
  unsigned &precedence)
{
  precedence=22;
  std::string dest;

  forall_operands(it, src)
  {
    if(it!=src.operands().begin()) dest+=", ";
    dest+=convert(*it);
  }

  return dest;
}

/*******************************************************************\

Function: expr2bpt::convert_sideeffect

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

std::string expr2bpt::convert_sideeffect(
  const exprt &src,
  unsigned &precedence)
{
  //const irep_idt &statement=src.get("statement");
  
  return convert_norep(src, precedence);
}

/*******************************************************************\

Function: expr2bpt::convert_code_function_call

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

std::string expr2bpt::convert_code_function_call(
  const code_function_callt &src)
{
  std::string result;

  if(src.lhs().is_not_nil())
  {
    result+=convert(src.lhs());
    result+=":=";
  }  
  
  result+=convert(src.function());
  
  result+="(";
  
  const exprt::operandst &arguments=src.arguments();
  
  forall_expr(it, arguments)
  {
    if(it!=arguments.begin()) result+=", ";
    result+=convert(*it);
  }
  
  result+=");";
  
  return result;
}

/*******************************************************************\

Function: expr2bpt::convert_next_symbol

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

std::string expr2bpt::convert_next_symbol(
  const exprt &src,
  unsigned &precedence)
{
  return "'"+convert_symbol(src, precedence);
}

/*******************************************************************\

Function: expr2bpt::convert_predicate_symbol

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

std::string expr2bpt::convert_predicate_symbol(
  const exprt &src,
  unsigned &precedence)
{
  precedence=22;
  return "P"+src.get_string("identifier");
}

/*******************************************************************\

Function: expr2bpt::convert_constant

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

std::string expr2bpt::convert_constant(
  const exprt &src,
  unsigned &precedence)
{
  precedence=22;

  const typet &type=src.type();

  if(type.id()=="bool")
  {
    if(src.is_true())
      return "1";
    else
      return "0";
  }
  else if(type.id()=="integer" ||
          type.id()=="natural" ||
          type.id()=="range")
    return src.get_string("value");

  return convert_norep(src, precedence);
}

/*******************************************************************\

Function: expr2bpt::convert_code_goto

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

std::string expr2bpt::convert_code_goto(const codet &src)
{
  return "goto ";
}

/*******************************************************************\

Function: expr2bpt::convert_code_ifthenelse

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

std::string expr2bpt::convert_code_ifthenelse(const codet &src)
{
  return "if ";
}

/*******************************************************************\

Function: expr2bpt::convert_code_assign

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

std::string expr2bpt::convert_code_assign(const code_assignt &src)
{
  return convert(src.lhs())+":="+convert(src.rhs())+";";
}

/*******************************************************************\

Function: expr2bpt::convert_code_bp_constrain

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

std::string expr2bpt::convert_code_bp_constrain(const codet &src)
{
  if(src.operands().size()!=2)
  {
    unsigned precedence;
    return convert_norep(src, precedence);
  }
  
  std::string result;

  if(src.op0().get("statement")=="assign")
  {
    result=convert(src.op0().op0())+":="+convert(src.op0().op1());
  }  
  
  result+=" constrain "+convert(src.op1())+";";
  
  return result;
}

/*******************************************************************\

Function: expr2bpt::convert_code_decl

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

std::string expr2bpt::convert_code_decl(const code_declt &src)
{
  if(src.operands().size()!=1)
  {
    unsigned precedence;
    return convert_norep(src, precedence);
  }
  
  return "decl "+convert(src.op0())+";";
}

/*******************************************************************\

Function: expr2bpt::convert_code_return

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

std::string expr2bpt::convert_code_return(const code_returnt &src)
{
  if(src.operands().size()!=1 &&
     src.operands().size()!=0)
  {
    unsigned precedence;
    return convert_norep(src, precedence);
  }

  if(src.operands().size()==1)
    return "return "+convert(src.op0())+";";
    
  return "return;";
}

/*******************************************************************\

Function: expr2bpt::convert_code

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

std::string expr2bpt::convert_code(const codet &src)
{
  const irep_idt &statement=src.get_statement();

  if(statement=="goto")
    return convert_code_goto(src);

  else if(statement=="ifthenelse")
    return convert_code_ifthenelse(src);

  else if(statement=="assign")
    return convert_code_assign(to_code_assign(src));

  else if(statement=="bp_constrain")
    return convert_code_bp_constrain(src);

  else if(statement=="function_call")
    return convert_code_function_call(to_code_function_call(src));

  else if(statement=="decl")
    return convert_code_decl(to_code_decl(src));

  else if(statement=="return")
    return convert_code_return(to_code_return(src));

  unsigned precedence;
  return convert_norep(src, precedence);
}

/*******************************************************************\

Function: expr2bpt::convert

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

std::string expr2bpt::convert(const exprt &src, unsigned &precedence)
{
  precedence=22;

  if(src.id()=="+")
    return convert_binary(src, "+", precedence=14);

  else if(src.id()=="-")
  {
    if(src.operands().size()<2)
      return convert_norep(src, precedence);
    else     
      return convert_binary(src, "-", precedence=14);
  }

  else if(src.id()=="unary-")
  {
    if(src.operands().size()!=1)
      return convert_norep(src, precedence);
    else     
      return convert_unary(src, "-", precedence=16);
  }

  else if(src.id()=="index")
    return convert_index(src, precedence=22);

  else if(src.id()=="*" || src.id()=="/")
    return convert_binary(src, src.id_string(), precedence=14);

  else if(src.id()=="<" || src.id()==">" ||
          src.id()=="<=" || src.id()==">=")
    return convert_binary(src, src.id_string(), precedence=9);

  else if(src.id()=="=")
    return convert_binary(src, "=", precedence=9);

  else if(src.id()=="not")
    return convert_unary(src, "!", precedence=16);

  else if(src.id()=="and")
    return convert_binary(src, "&", precedence=7);

  else if(src.id()=="or")
    return convert_binary(src, "|", precedence=6);

  else if(src.id()=="=>")
    return convert_implies(src, precedence=6);

  else if(src.id()=="AG" || src.id()=="EG" ||
          src.id()=="AX" || src.id()=="EX")
    return convert_unary(src, src.id_string()+" ", precedence=4);

  else if(src.id()=="symbol")
    return convert_symbol(src, precedence);

  else if(src.id()=="bool-vector")
    return convert_bool_vector(src, precedence);

  else if(src.id()=="predicate_symbol")
    return convert_predicate_symbol(src, precedence);

  else if(src.id()=="next_symbol")
    return convert_next_symbol(src, precedence);

  else if(src.id()=="sideeffect")
    return convert_sideeffect(src, precedence);

  else if(src.id()=="constant")
    return convert_constant(src, precedence);

  else if(src.id()=="bp_unused")
    return "_";

  else if(src.id()=="nondet_bool")
    return convert_nondet_bool(src, precedence);
    
  else if(src.id()=="code")
    return convert_code(to_code(src));

  // no Boolean Program language expression for internal representation 
  return convert_norep(src, precedence);
}

/*******************************************************************\

Function: expr2bpt::convert

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

std::string expr2bpt::convert(const exprt &src)
{
  unsigned precedence;
  return convert(src, precedence);
}

/*******************************************************************\

Function: expr2bp

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

std::string expr2bp(const exprt &expr)
{
  expr2bpt expr2bp;
  return expr2bp.convert(expr);
}

/*******************************************************************\

Function: type2bp

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

std::string type2bp(const typet &type)
{
  if(type.id()=="bool")
    return "bool";
  else if(type.id()=="empty")
    return "void";
  else if(type.id()=="bool-vector")
    return "bool<"+type.get_string("width")+">";

  return "unknown-type";
}
