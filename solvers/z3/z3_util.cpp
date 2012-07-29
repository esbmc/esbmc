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

  z3::expr expr = z3::to_expr(*ctx, formula);
  z3::expr sym = ctx->constant(the_name.c_str(), expr.get_sort());
  z3::expr eq = sym == expr;
  assert_formula(eq);
  return;
}
