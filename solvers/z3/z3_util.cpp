/*******************************************************************\

Module:

Author: Lucas Cordeiro, lcc08r@ecs.soton.ac.uk

\*******************************************************************/

#include <sstream>
#include <iomanip>

#include "z3_conv.h"

std::string
z3_convt::double2string(double d) const
{
  std::ostringstream format_message;
  format_message << std::setprecision(12) << d;
  return format_message.str();
}

std::string
z3_convt::itos(long int i)
{
  std::stringstream s;
  s << i;
  return s.str();
}

const expr2tc
z3_convt::label_formula(std::string name, const type2tc &t,
                        const z3::expr &formula)
{
  unsigned &num = label_map.back()[name], level;
  std::string basename = "__ESBMC_z3_label_map_" + name;
  std::string the_name = basename + "&0#" + itos(num);
  level = num;
  num++;

  z3::expr sym = ctx.constant(the_name.c_str(), formula.get_sort());
  z3::expr eq = sym == formula;
  assert_formula(eq);

  return symbol2tc(t, basename, symbol2t::level2_global, 0, level, 0, 0);
}
