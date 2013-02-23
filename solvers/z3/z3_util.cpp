/*******************************************************************\

Module:

Author: Lucas Cordeiro, lcc08r@ecs.soton.ac.uk

\*******************************************************************/

#include <sstream>

#include "z3_conv.h"

void
z3_convt::debug_label_formula(std::string name, const z3::expr &formula)
{
  std::stringstream ss;
  unsigned &num = debug_label_map[name];
  ss << "__ESBMC_" << name << num;
  std::string the_name = ss.str();
  num++;

  z3::expr sym = ctx.constant(the_name.c_str(), formula.get_sort());
  z3::expr eq = sym == formula;
  assert_formula(eq);
  return;
}
