/*******************************************************************\

Module:

Author: Lucas Cordeiro, lcc08r@ecs.soton.ac.uk

\*******************************************************************/

#include <sstream>

#include "z3_conv.h"

std::string z3_convt::itos(int i)
{
  std::stringstream s;
  s << i;

  return s.str();
}

void
z3_convt::debug_label_formula(std::string name, Z3_ast formula)
{
  unsigned &num = debug_label_map[name];
  std::string the_name = "__ESBMC_" + name + itos(num);
  num++;

  Z3_sort sort = Z3_get_sort(z3_ctx, formula);
  Z3_ast sym = z3_api.mk_var(the_name.c_str(), sort);
  Z3_ast eq = Z3_mk_eq(z3_ctx, sym, formula);
  assert_formula(eq);
  return;
}
