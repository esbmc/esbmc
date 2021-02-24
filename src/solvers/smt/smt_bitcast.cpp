#include <solvers/smt/smt_conv.h>

smt_astt smt_convt::convert_bitcast(const expr2tc &expr)
{
  assert(is_bitcast2t(expr));

  const expr2tc &from = to_bitcast2t(expr).from;
  const type2tc &to_type = to_bitcast2t(expr).type;

  // Converts from bitvector to floating-point
  if(is_floatbv_type(to_type) && is_bv_type(from))
    return fp_api->mk_from_bv_to_fp(convert_ast(from), convert_sort(to_type));

  if(is_floatbv_type(from) && is_bv_type(to_type))
  {
    unsigned int sz = expr->type->get_width() - from->type->get_width();
    return is_signedbv_type(expr->type)
             ? mk_sign_ext(fp_api->mk_from_fp_to_bv(convert_ast(from)), sz)
             : mk_zero_ext(fp_api->mk_from_fp_to_bv(convert_ast(from)), sz);
  }

  // Cast by value is fine
  return convert_ast(typecast2tc(to_type, from));
}
