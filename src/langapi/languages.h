/*******************************************************************\

Module:

Author: Daniel Kroening, kroening@cs.cmu.edu

\*******************************************************************/

#ifndef CPROVER_LANGUAGES_H
#define CPROVER_LANGUAGES_H

#include <util/language.h>

class languagest
{
public:
  // conversion of expressions

  bool from_expr(const exprt &expr, std::string &code)
  {
    return language->from_expr(expr, code, ns);
  }

  bool from_type(const typet &type, std::string &code)
  {
    return language->from_type(type, code, ns);
  }

  // constructor / destructor

  languagest(const namespacet &_ns, const char *mode, const messaget &msg);
  virtual ~languagest();

protected:
  const namespacet &ns;
  languaget *language;
  const messaget &msg;
};

#endif
