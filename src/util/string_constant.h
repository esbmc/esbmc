#ifndef CPROVER_ANSI_C_EXPR_H
#define CPROVER_ANSI_C_EXPR_H

#include <util/expr.h>

class string_constantt : public exprt
{
public:
  explicit string_constantt(const irep_idt &value);
  explicit string_constantt(
    const irep_idt &value,
    const typet &type,
    bool is_wide);

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

  irep_idt mb_value() const;

  class mb_conversion_error : public std::runtime_error
  {
    using std::runtime_error::runtime_error;
  };
};

const string_constantt &to_string_constant(const exprt &expr);
string_constantt &to_string_constant(exprt &expr);

#endif
