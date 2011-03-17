/*******************************************************************\

Module:

Author: Daniel Kroening, kroening@cs.cmu.edu

\*******************************************************************/

#include <string.h>

#include "expr2cvc.h"
#include "lispexpr.h"
#include "lispirep.h"

class expr2cvct
{
 public:
  bool convert_binary(const exprt &src, std::string &dest, const std::string &symbol,
                      unsigned precedence);

  bool convert_unary(const exprt &src, std::string &dest, const std::string &symbol,
                     unsigned precedence);

  bool convert_index(const exprt &src, std::string &dest, unsigned precedence);

  bool convert(const exprt &src, std::string &dest, unsigned &precedence);

  bool convert(const exprt &src, std::string &dest);

  bool convert_symbol(const exprt &src, std::string &dest, unsigned &precedence);

  bool convert_constant(const exprt &src, std::string &dest, unsigned &precedence);

  bool convert_norep(const exprt &src, std::string &dest, unsigned &precedence);

  bool convert(const typet &src, std::string &dest);
};

/*******************************************************************\

Function: expr2cvct::convert_binary

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

bool expr2cvct::convert_binary(const exprt &src, std::string &dest,
                               const std::string &symbol,
                               unsigned precedence)
{
  if(src.operands().size()<2)
    return convert_norep(src, dest, precedence);

  bool first=true;

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

    std::string op;
    unsigned p;

    if(convert(*it, op, p)) return true;

    if(precedence>p) dest+='(';
    dest+=op;
    if(precedence>p) dest+=')';
  }

  return false;
}

/*******************************************************************\

Function: expr2cvct::convert_unary

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

bool expr2cvct::convert_unary(const exprt &src, std::string &dest,
                              const std::string &symbol,
                              unsigned precedence)
{
  if(src.operands().size()!=1)
    return convert_norep(src, dest, precedence);
    
  std::string op;
  unsigned p;

  if(convert(src.op0(), op, p)) return true;

  dest+=symbol;
  if(precedence>p) dest+='(';
  dest+=op;
  if(precedence>p) dest+=')';

  return false;
}

/*******************************************************************\

Function: expr2cvct::convert_index

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

bool expr2cvct::convert_index(const exprt &src, std::string &dest,
                              unsigned precedence)
{
  if(src.operands().size()!=2)
    return convert_norep(src, dest, precedence);

  std::string op;
  unsigned p;

  if(convert(src.op0(), op, p)) return true;

  if(precedence>p) dest+='(';
  dest+=op;
  if(precedence>p) dest+=')';

  if(convert(src.op1(), op, p)) return true;

  dest+='[';
  dest+=op;
  dest+=']';

  return false;
}

/*******************************************************************\

Function: expr2cvct::convert_norep

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

bool expr2cvct::convert_norep(const exprt &src, std::string &dest,
                              unsigned &precedence)
{
  precedence=22;
  dest=src.to_string();
  return false;
}

/*******************************************************************\

Function: expr2cvct::convert_symbol

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

bool expr2cvct::convert_symbol(const exprt &src, std::string &dest,
                               unsigned &precedence)
{
  precedence=22;
  dest=src.get_string("identifier");
 
  if(strncmp(dest.c_str(), "cvc::", 5)==0)
    dest.erase(0, 5);

  return false;
}

/*******************************************************************\

Function: expr2cvct::convert_constant

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

bool expr2cvct::convert_constant(const exprt &src, std::string &dest,
                                 unsigned &precedence)
{
  precedence=22;

  const typet &type=src.type();
  const std::string &value=src.get_string("value");

  if(type.id()=="bool")
  {
    if(src.is_true())
      dest+="1";
    else
      dest+="0";
  }
  else if(type.id()=="integer" || type.id()=="natural")
    dest=value;
  else
    return convert_norep(src, dest, precedence);

  return false;
}

/*******************************************************************\

Function: expr2cvct::convert

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

bool expr2cvct::convert(const exprt &src, std::string &dest, unsigned &precedence)
{
  precedence=22;

  if(src.id()=="+")
    return convert_binary(src, dest, "+", precedence=14);

  else if(src.id()=="-")
  {
    if(src.operands().size()<2)
      return convert_norep(src, dest, precedence);
    else     
      return convert_binary(src, dest, "-", precedence=14);
  }

  else if(src.id()=="unary-")
  {
    if(src.operands().size()!=1)
      return convert_norep(src, dest, precedence);
    else     
      return convert_unary(src, dest, "-", precedence=16);
  }

  else if(src.id()=="index")
    return convert_index(src, dest, precedence=22);

  else if(src.id()=="*" || src.id()=="/")
    return convert_binary(src, dest, src.id_string(), precedence=14);

  else if(src.id()=="<" || src.id()==">" ||
          src.id()=="<=" || src.id()==">=")
    return convert_binary(src, dest, src.id_string(), precedence=9);

  else if(src.id()=="=")
    return convert_binary(src, dest, "=", precedence=9);

  else if(src.id()=="not")
    return convert_unary(src, dest, "!", precedence=16);

  else if(src.id()=="and")
    return convert_binary(src, dest, "&", precedence=7);

  else if(src.id()=="or")
    return convert_binary(src, dest, "|", precedence=6);

  else if(src.id()=="=>")
    return convert_binary(src, dest, "->", precedence=5);

  else if(src.id()=="<=>")
    return convert_binary(src, dest, "<->", precedence=4);

  else if(src.id()=="symbol")
    return convert_symbol(src, dest, precedence);

  else if(src.id()=="constant")
    return convert_constant(src, dest, precedence);

  else // no SMV language expression for internal representation 
    return convert_norep(src, dest, precedence);

  return false;
}

/*******************************************************************\

Function: expr2cvct::convert

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

bool expr2cvct::convert(const exprt &src, std::string &dest)
{
  unsigned precedence;
  return convert(src, dest, precedence);
}

/*******************************************************************\

Function: expr2cvct::convert

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

bool expr2cvct::convert(const typet &src, std::string &dest)
{
  // not done yet
  return true;
}

/*******************************************************************\

Function: expr2cvc

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

bool expr2cvc(const exprt &expr, std::string &code)
{
  expr2cvct expr2cvc;
  return expr2cvc.convert(expr, code);
}

/*******************************************************************\

Function: type2cvc

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

bool type2cvc(const typet &type, std::string &code)
{
  expr2cvct expr2cvc;
  return expr2cvc.convert(type, code);
}

