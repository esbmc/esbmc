/*******************************************************************\

Module: Misc Utilities

Author: Daniel Kroening, kroening@kroening.com

\*******************************************************************/

#include <util/array_name.h>

std::string array_name(
  const namespacet &ns,
  const exprt &expr)
{
  if(expr.id()=="index")
  {
    if(expr.operands().size()!=2)
      throw "index takes two operands";

    return array_name(ns, expr.op0())+"[]";
  }
  else if(expr.id()=="symbol")
  {
    const symbolt &symbol=ns.lookup(expr);
    return "array `"+id2string(symbol.base_name)+"'";
  }
  else if(expr.id()=="string-constant")
  {
    return "string";
  }

  return "array";
}

std::string array_name(
  const namespacet &ns,
  const expr2tc &expr)
{
  if (is_index2t(expr))
  {
    return array_name(ns, to_index2t(expr).source_value) + "[]";
  }
  else if (is_symbol2t(expr))
  {
    const symbolt &symbol=ns.lookup(to_symbol2t(expr).thename);
    return "array `"+id2string(symbol.base_name)+"'";
  }
  else if (is_constant_string2t(expr))
  {
    return "string";
  }

  return "array";
}

