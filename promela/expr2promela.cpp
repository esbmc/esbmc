/*******************************************************************\

Module:

Author: Daniel Kroening, kroening@kroening.com

\*******************************************************************/

#include "expr2promela.h"

#include <ansi-c/expr2c.h>

class expr2promelat:public expr2ct
{
 public:
  expr2promelat(const namespacet &_ns):expr2ct(_ns) { }

  std::string convert(const typet &src)
  {
    return expr2ct::convert(src);
  }

  std::string convert(const exprt &src)
  {
    return expr2ct::convert(src);
  }
  
 protected:
  //virtual bool convert_code(const exprt &src, std::string &dest);
  //virtual bool convert_code(const exprt &src, std::string &dest, unsigned indent);

  virtual std::string convert(const exprt &src, unsigned &precedence);
  //virtual bool convert_symbol(const exprt &src, std::string &dest, unsigned &precedence);
  // virtual bool convert_constant(const exprt &src, std::string &dest, unsigned &precedence);
};

/*******************************************************************\

Function: expr2ct::convert

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

std::string expr2promelat::convert(
  const exprt &src,
  unsigned &precedence)
{
  return expr2ct::convert(src, precedence);
}

/*******************************************************************\

Function: expr2promela

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

std::string expr2promela(const exprt &expr, const namespacet &ns)
{
  expr2promelat expr2promela(ns);
  expr2promela.get_shorthands(expr);
  return expr2promela.convert(expr);
}

/*******************************************************************\

Function: type2promela

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

std::string type2promela(const typet &type, const namespacet &ns)
{
  expr2promelat expr2promela(ns);
  return expr2promela.convert(type);
}
