/*******************************************************************\

Module:

Author: Daniel Kroening, kroening@kroening.com

\*******************************************************************/

#include "language.h"

bool languaget::from_expr(
  const exprt &expr,
  std::string &code,
  const namespacet &ns __attribute__((unused)))
{
  code=expr.pretty();
  return false;
}

bool languaget::from_type(
  const typet &type,
  std::string &code,
  const namespacet &ns __attribute__((unused)))
{
  code=type.pretty();
  return false;
}
