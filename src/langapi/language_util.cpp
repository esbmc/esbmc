/*******************************************************************\

Module:

Author: Daniel Kroening, kroening@cs.cmu.edu

\*******************************************************************/

#include <langapi/language_util.h>
#include <langapi/mode.h>
#include <memory>

/*******************************************************************\

Function: from_expr

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

std::string from_expr(
  const namespacet &ns,
  const irep_idt &identifier,
  const exprt &expr)
{
  int mode;

  if(identifier=="")
    mode=0;
  else
  {
    const symbolt *symbol;

    if(ns.lookup(identifier, symbol))
      mode=0;
    else if(symbol->mode=="")
      mode=0;
    else
    {
      mode=get_mode(id2string(symbol->mode));
      if(mode<0)
        throw "symbol "+id2string(symbol->name)+
              " has unknown mode '"+id2string(symbol->mode)+"'";
    }
  }

  std::unique_ptr<languaget> p(mode_table[mode].new_language());
  std::string result;
  p->from_expr(expr, result, ns);
  return result;
}

/*******************************************************************\

Function: from_type

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

std::string from_type(
  const namespacet &ns,
  const irep_idt &identifier,
  const typet &type)
{
  int mode;

  if(identifier=="")
    mode=0;
  else
  {
    const symbolt *symbol;

    if(ns.lookup(identifier, symbol))
      mode=0;
    else if(symbol->mode=="")
      mode=0;
    else
    {
      mode=get_mode(id2string(symbol->mode));
      if(mode<0)
        throw "symbol "+id2string(symbol->name)+
              " has unknown mode '"+id2string(symbol->mode)+"'";
    }
  }

  std::unique_ptr<languaget> p(mode_table[mode].new_language());
  std::string result;
  p->from_type(type, result, ns);
  return result;
}

/*******************************************************************\

Function: from_expr

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

std::string from_expr(const exprt &expr)
{
  contextt context;
  return from_expr(namespacet(context), "", expr);
}

/*******************************************************************\

Function: from_type

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

std::string from_type(const typet &type)
{
  contextt context;
  return from_type(namespacet(context), "", type);
}

