
#pragma once

#include <map>
#include <set>
#include <util/c_qualifiers.h>
#include <util/expr.h>
#include <util/namespace.h>
#include <util/std_code.h>
#include <util/c_expr2string.h>

std::string expr2c(const exprt &expr, const namespacet &ns, unsigned flags = 0);
std::string type2c(const typet &type, const namespacet &ns, unsigned flags = 0);
std::string
typedef2c(const typet &type, const namespacet &ns, unsigned flags = 0);

class expr2ct : public c_expr2stringt
{
public:
  using c_expr2stringt::flagst;

  expr2ct(const namespacet &_ns, unsigned _flags) : c_expr2stringt(_ns, _flags)
  {
  }
  virtual ~expr2ct() = default;

  std::string convert(const typet &src) override;
  std::string convert(const exprt &src) override;
  std::string convert_struct_union_typedef(const struct_union_typet &src);

  // Some static methods that we should probably redesign in the future
  static std::string get_name_shorthand(std::string fullname);
  static bool is_padding(std::string tag);
  static bool is_anonymous_member(std::string tag);
  static bool is_typedef_struct_union(std::string tag);

protected:
  std::string convert_rec(
    const typet &src,
    const c_qualifierst &qualifiers,
    const std::string &declarator) override;

  std::string convert_struct_union_body(
    const exprt &src,
    const exprt::operandst &operands,
    const struct_union_typet::componentst &components) override;
  std::string convert_struct(const exprt &src, unsigned &precedence) override;
  std::string convert_union(const exprt &src, unsigned &precedence) override;

  std::string convert_typecast(const exprt &src, unsigned &precedence) override;

  std::string convert_code_printf(const codet &src, unsigned indent) override;
  std::string convert_code_free(const codet &src, unsigned indent) override;
  std::string convert_code_return(const codet &src, unsigned indent) override;
  std::string convert_code_assign(const codet &src, unsigned indent) override;
  std::string convert_code_assume(const codet &src, unsigned indent) override;
  std::string convert_code_assert(const codet &src, unsigned indent) override;

  std::string convert_malloc(const exprt &src, unsigned &precedence) override;
  std::string convert_realloc(const exprt &src, unsigned &precedence) override;
  std::string convert_alloca(const exprt &src, unsigned &precedence) override;
  std::string convert_symbol(const exprt &src, unsigned &precedence) override;
  std::string convert_constant(const exprt &src, unsigned &precedence) override;

  std::string convert_member(const exprt &src, unsigned precedence) override;

  std::string convert_code_decl(const codet &src, unsigned indent) override;
  std::string convert(const exprt &src, unsigned &precedence) override;

  std::string convert_same_object(const exprt &src, unsigned &precedence);
  std::string convert_pointer_offset(
    const exprt &src,
    unsigned &precedence [[maybe_unused]]);
  std::string convert_infinity(
    const exprt &src [[maybe_unused]],
    unsigned &precedence [[maybe_unused]]);
  std::string convert_dynamic_size(const exprt &src, unsigned &precedence);

  std::string convert_ieee_div(const exprt &src, unsigned &precedence);
  std::string convert_ieee_mul(const exprt &src, unsigned &precedence);
  std::string convert_ieee_add(const exprt &src, unsigned &precedence);
  std::string convert_ieee_sub(const exprt &src, unsigned &precedence);
  std::string convert_ieee_sqrt(const exprt &src, unsigned &precedence);

  std::string convert_nondet(const exprt &src, unsigned &precedence) override;
  std::string convert_array_of(const exprt &src, unsigned precedence) override;

private:
};
