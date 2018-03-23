/*******************************************************************\

Module:

Author: Daniel Kroening, kroening@kroening.com

\*******************************************************************/

#ifndef CPROVER_ANSI_C_EXPR_H
#define CPROVER_ANSI_C_EXPR_H

#include <util/expr.h>

class string_constantt : public exprt
{
public:
  explicit string_constantt(const irep_idt &value);
  explicit string_constantt(const irep_idt &value, const typet &type);

  friend inline const string_constantt &to_string_constant(const exprt &expr)
  {
    assert(expr.id() == "string-constant");
    return static_cast<const string_constantt &>(expr);
  }

  friend inline string_constantt &to_string_constant(exprt &expr)
  {
    assert(expr.id() == "string-constant");
    return static_cast<string_constantt &>(expr);
  }

  void set_value(const irep_idt &value);

  const irep_idt &get_value() const
  {
    return value();
  }
};

const string_constantt &to_string_constant(const exprt &expr);
string_constantt &to_string_constant(exprt &expr);

#endif
