/*******************************************************************\

Module:

Author: Daniel Kroening, kroening@kroening.com

\*******************************************************************/

#ifndef CPROVER_EXPR2CCODE_H
#define CPROVER_EXPR2CCODE_H

#include <map>
#include <set>
#include <util/c_qualifiers.h>
#include <util/expr.h>
#include <util/namespace.h>
#include <util/std_code.h>
#include <clang-c-frontend/expr2c.h>

std::string
expr2ccode(const exprt &expr, const namespacet &ns, bool fullname = false);

std::string
type2ccode(const typet &type, const namespacet &ns, bool fullname = false);

class expr2ccodet : public expr2ct
{
public:
  expr2ccodet(const namespacet &_ns, const bool _fullname)
    : expr2ct(_ns, _fullname)
  {
  }
  virtual ~expr2ccodet() = default;

  std::string convert(const typet &src) override;
  std::string convert(const exprt &src) override;
  
  // Static list to keep track of the global compound types that have
  // already been declared declared
  static std::list<std::string> declared_types;
  
  // A method for removing a type from the list of declared types
  static void remove_declared_type(std::string type);


protected:

  std::string convert_rec(
    const typet &src,
    const c_qualifierst &qualifiers,
    const std::string &declarator) override;

  std::string convert_struct_union_body(
    const exprt &src,
    const exprt::operandst &operands,
    const struct_union_typet::componentst &components);
  std::string convert_struct(const exprt &src, unsigned &precedence) override;
  std::string convert_union(const exprt &src, unsigned &precedence) override;

  std::string convert_typecast(const exprt &src, unsigned &precedence) override;
  std::string convert_struct_typedef(const typet &src);
  std::string convert_union_typedef(const typet &src);
  std::string convert_struct_union_typedef(const typet &src);

  std::string convert_code_printf(const codet &src, unsigned indent) override;
  std::string convert_code_free(const codet &src, unsigned indent) override;
  std::string convert_malloc(const exprt &src, unsigned &precedence) override;
  std::string convert_alloca(const exprt &src, unsigned &precedence) override;
  std::string convert_symbol(const exprt &src, unsigned &precedence) override;

  std::string convert_member(const exprt &src, unsigned precedence) override;

  std::string convert_code_decl(const codet &src, unsigned indent) override;
  std::string convert(const exprt &src, unsigned &precedence) override;

  std::string convert_same_object(const exprt &src, unsigned &precedence);
  std::string convert_pointer_offset(const exprt &src, unsigned &precedence);
  std::string convert_infinity(const exprt &src, unsigned &precedence);
  std::string convert_dynamic_size(const exprt &src, unsigned &precedence);

  std::string convert_ieee_div(const exprt &src, unsigned &precedence);
  std::string convert_ieee_mul(const exprt &src, unsigned &precedence);
  std::string convert_ieee_add(const exprt &src, unsigned &precedence);
  std::string convert_ieee_sub(const exprt &src, unsigned &precedence);

  std::string convert_nondet(const exprt &src, unsigned &precedence) override;

  std::string convert_array_of(const exprt &src, unsigned precedence) override;

private:
  std::string convert_from_ssa_form(const std::string symbol);
};

#endif
