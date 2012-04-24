/*******************************************************************\

Module:

Author: Lucas Cordeiro, lcc08r@ecs.soton.ac.uk

\*******************************************************************/

#include <sstream>

#include "z3_conv.h"

Z3_ast z3_convt::convert_number(int64_t value, u_int width, bool type)
{

  if (int_encoding)
    return convert_number_int(value, width, type);
  else
    return convert_number_bv(value, width, type);
}

Z3_ast z3_convt::convert_number_int(int64_t value, u_int width, bool type)
{

  if (type)
    return z3_api.mk_int(value);
  else
    return z3_api.mk_unsigned_int(value);
}

Z3_ast z3_convt::convert_number_bv(int64_t value, u_int width, bool type)
{

  if (type)
    return Z3_mk_int(z3_ctx, value, Z3_mk_bv_type(z3_ctx, width));
  else
    return Z3_mk_unsigned_int(z3_ctx, value, Z3_mk_bv_type(z3_ctx, width));
}

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
