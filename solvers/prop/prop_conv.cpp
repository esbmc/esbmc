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

  cachet::iterator it = cache.find(expr);
  if (it != cache.end())
    return it->l;

  literalt literal = convert_expr(expr);

  // insert into cache

  struct lit_cachet entry = { expr, literal, 0 };
  cache.insert(entry);

  return literal;
}

void prop_convt::ignoring(const expr2tc &expr)
{
  // fall through

  std::string msg="warning: ignoring "+expr->pretty();

  print(2, msg);
}

void prop_convt::convert_smt_type(const type2t &type,
                                  void *&arg __attribute__((unused)))
{
  std::cerr << "Unhandled SMT conversion for type \""
            << get_type_id(type) << std::endl;
  abort();
}

void prop_convt::convert_smt_expr(const expr2t &expr,
                                  void *&arg __attribute__((unused)))
{
  std::cerr << "Unhandled SMT conversion for expr ID "
            << get_expr_id(expr) << std::endl;
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

void prop_convt::push_ctx(void)
{
  ctx_level++;
}

void prop_convt::pop_ctx(void)
{
  cachet::nth_index<1>::type &cache_numindex = cache.get<1>();
  cache_numindex.erase(ctx_level);

  ctx_level--;
}
