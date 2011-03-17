/*******************************************************************\

Module: SMT-LIB Frontend, expression conversion

Author: CM Wintersteiger

\*******************************************************************/

#include "expr2smt.h"

/*******************************************************************\

Function: expr2smt

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

bool expr2smt(const exprt &expr, std::string &code)
{
  /* TODO */
  code = expr.id_string();
  return false;
}

/*******************************************************************\

Function: type2smt

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

bool type2smt(const typet &type, std::string &code)
{
  /* TODO */
  code = type.id_string();  
  return false;
}

/*******************************************************************\

Function: ii2string

  Inputs:

 Outputs:

 Purpose: converts an indexed ident into a string

\*******************************************************************/

std::string ii2string(const irept &ident)
{
  std::string res = ident.id_string();
  const exprt &e = static_cast<const exprt&>(ident);
  const exprt &i = static_cast<const exprt&>(e.find("index"));
  if (i.is_not_nil())
  {
    forall_operands(it, i)
    {
      if (it->id()=="index")    
        res += "["+it->get_string("value")+"]";
    }
  }
  return res;
}
