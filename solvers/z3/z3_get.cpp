/*******************************************************************\

   Module:

   Author: Lucas Cordeiro, lcc08r@ecs.soton.ac.uk

\*******************************************************************/

#include <assert.h>

#include <iostream>
#include <math.h>
#include <migrate.h>
#include <iomanip>
#include <sstream>
#include <string>

#include <arith_tools.h>
#include <std_expr.h>
#include <std_types.h>

#include "z3_conv.h"

std::string
z3_convt::double2string(double d) const
{

  std::ostringstream format_message;
  format_message << std::setprecision(12) << d;
  return format_message.str();
}

expr2tc
z3_convt::get_bool(const smt_ast *a)
{
  assert(a->sort->id == SMT_SORT_BOOL);
  const z3_smt_ast *za = z3_smt_downcast(a);

  z3::expr e = za->e;
  try {
    e = model.eval(e, false);
  } catch (z3::exception &e) {
    // No model value
    return expr2tc();
  }

  if (Z3_get_bool_value(z3_ctx, e) == Z3_L_TRUE)
    return true_expr;
  else
    return false_expr;
}

expr2tc
z3_convt::get_bv(const type2tc &t, const smt_ast *a)
{
  const z3_smt_ast *za = z3_smt_downcast(a);

  z3::expr e = za->e;
  try {
    e = model.eval(e, false);
  } catch (z3::exception &e) {
    // No model value
    return expr2tc();
  }

  if (Z3_get_ast_kind(z3_ctx, e) != Z3_NUMERAL_AST)
    return expr2tc();

  std::string value = Z3_get_numeral_string(z3_ctx, e);
  return constant_int2tc(t, BigInt(value.c_str()));
}

expr2tc
z3_convt::get_array_elem(const smt_ast *array, uint64_t index,
                         const smt_sort *elem_sort)
{
  const z3_smt_ast *za = z3_smt_downcast(array);
  unsigned long bv_size = array->sort->get_domain_width();
  z3_smt_ast *idx =
    static_cast<z3_smt_ast*>(mk_smt_bvint(BigInt(index), false, bv_size));

  z3::expr e = select(za->e, idx->e);
  delete idx;
  try {
    e = model.eval(e, false);
  } catch (z3::exception &e) {
    // No model value
    return expr2tc();
  }

  z3_smt_ast *value = new z3_smt_ast(e, elem_sort);
  type2tc res_type = get_uint_type(bv_size);
  expr2tc result = get_bv(res_type, value);
  delete value;

  return result;
}
