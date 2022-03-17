
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
  expr2ccodet(const namespacet &_ns, const bool _fullname) : expr2ct(_ns, _fullname)
  {
  }
  virtual ~expr2ccodet() = default;
  
  std::string convert(const typet &src) override;
  std::string convert(const exprt &src) override;

protected:

  std::string convert_rec(
    const typet &src,
    const c_qualifierst &qualifiers,
    const std::string &declarator) override;

  std::string convert_code_printf(const codet &src, unsigned indent) override;
  std::string convert_malloc(const exprt &src, unsigned &precedence) override;
  std::string convert_symbol(const exprt &src, unsigned &precedence) override;
  
  std::string convert(const exprt &src, unsigned &precedence) override;

  std::string convert_same_object(const exprt &src, unsigned &precedence);

private:
  std::string convert_from_ssa_form(const std::string symbol);

};


#endif
