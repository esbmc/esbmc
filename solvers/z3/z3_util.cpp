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

const expr2tc
z3_convt::label_formula(std::string name, const type2tc &t,
                        const z3::expr &formula)
{
  unsigned &num = label_map.back()[name];
  std::string basename = "__ESBMC_z3_label_map_" + name;
  std::string the_name = basename + "&0#" + itos(num);
  num++;

  z3::expr sym = ctx.constant(the_name.c_str(), formula.get_sort());
  z3::expr eq = sym == formula;
  assert_formula(eq);

  return expr2tc(new symbol2t(t, basename, symbol2t::level2_global,
                              0, num, 0, 0));
}
