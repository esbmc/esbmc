/*******************************************************************\

Module:

Author: Daniel Kroening, kroening@kroening.com

\*******************************************************************/

#include <assert.h>

#include <map>

#include <irep2.h>
#include <migrate.h>
#include "prop_conv.h"

literalt prop_convt::convert(const expr2tc &expr)
{

  std::pair<cachet::iterator, bool> result=
    cache.insert(std::pair<expr2tc, literalt>(expr, literalt()));

  if(!result.second)
    return result.first->second;

  literalt literal = convert_expr(expr);

  // insert into cache

  result.first->second=literal;

  return literal;
}

void prop_convt::ignoring(const expr2tc &expr)
{
  // fall through

  std::string msg="warning: ignoring "+expr->pretty();

  print(2, msg);
}

void prop_convt::convert_smt_type(const type2t &type, void *&arg)
{
  std::cerr << "Unhandled SMT conversion for type ID " << type.type_id <<
               std::endl;
  abort();
}

void prop_convt::convert_smt_expr(const expr2t &expr, void *&arg)
{
  std::cerr << "Unhandled SMT conversion for expr ID " << expr.expr_id <<
               std::endl;
  abort();
}

void prop_convt::set_equal(literalt a, literalt b)
{
  bvt bv;
  bv.resize(2);
  bv[0]=a;
  bv[1]=lnot(b);
  lcnf(bv);
  bv[0]=lnot(a);
  bv[1]=b;
  lcnf(bv);
}
