#include <solvers/smt/smt_conv.h>

smt_astt smt_convt::mk_smt_nearbyint_from_float(expr2tc from, expr2tc rm)
{
  smt_astt _from = convert_ast(from);
  smt_astt _rm = convert_rounding_mode(rm);
  return fp_api->mk_smt_nearbyint_from_float(_from, _rm);
}

smt_astt smt_convt::mk_smt_fpbv_sqrt(expr2tc rd, expr2tc rm)
{
  smt_astt _rd = convert_ast(rd);
  smt_astt _rm = convert_rounding_mode(rm);
  return fp_api->mk_smt_fpbv_sqrt(_rd, _rm);
}

smt_astt
smt_convt::mk_smt_fpbv_fma(expr2tc v1, expr2tc v2, expr2tc v3, expr2tc rm)
{
  smt_astt _v1 = convert_ast(v1);
  smt_astt _v2 = convert_ast(v2);
  smt_astt _v3 = convert_ast(v3);
  smt_astt _rm = convert_rounding_mode(rm);
  return fp_api->mk_smt_fpbv_fma(_v1, _v2, _v3, _rm);
}

smt_astt
smt_convt::mk_smt_typecast_from_fpbv_to_ubv(expr2tc from, std::size_t width)
{
  smt_astt _from = convert_ast(from);
  smt_sortt _to = mk_int_bv_sort(SMT_SORT_SBV, width);
  return fp_api->mk_smt_typecast_from_fpbv_to_sbv(_from, _to);
}

smt_astt
smt_convt::mk_smt_typecast_from_fpbv_to_sbv(expr2tc from, std::size_t width)
{
  smt_astt _from = convert_ast(from);
  smt_sortt _to = mk_int_bv_sort(SMT_SORT_SBV, width);
  return fp_api->mk_smt_typecast_from_fpbv_to_sbv(_from, _to);
}

smt_astt smt_convt::mk_smt_typecast_from_fpbv_to_fpbv(
  expr2tc from,
  type2tc to,
  expr2tc rm)
{
  smt_astt _from = convert_ast(from);
  smt_sortt _to = convert_sort(to);
  smt_astt _rm = convert_rounding_mode(rm);
  return fp_api->mk_smt_typecast_from_fpbv_to_fpbv(_from, _to, _rm);
}

smt_astt
smt_convt::mk_smt_typecast_ubv_to_fpbv(expr2tc from, type2tc to, expr2tc rm)
{
  smt_astt _from = convert_ast(from);
  smt_sortt _to = convert_sort(to);
  smt_astt _rm = convert_rounding_mode(rm);
  return fp_api->mk_smt_typecast_ubv_to_fpbv(_from, _to, _rm);
}

smt_astt
smt_convt::mk_smt_typecast_sbv_to_fpbv(expr2tc from, type2tc to, expr2tc rm)
{
  smt_astt _from = convert_ast(from);
  smt_sortt _to = convert_sort(to);
  smt_astt _rm = convert_rounding_mode(rm);
  return fp_api->mk_smt_typecast_sbv_to_fpbv(_from, _to, _rm);
}

smt_astt smt_convt::mk_smt_fpbv_add(expr2tc lhs, expr2tc rhs, expr2tc rm)
{
  smt_astt _lhs = convert_ast(lhs);
  smt_astt _rhs = convert_ast(rhs);
  smt_astt _rm = convert_rounding_mode(rm);
  return fp_api->mk_smt_fpbv_add(_lhs, _rhs, _rm);
}

smt_astt smt_convt::mk_smt_fpbv_sub(expr2tc lhs, expr2tc rhs, expr2tc rm)
{
  smt_astt _lhs = convert_ast(lhs);
  smt_astt _rhs = convert_ast(rhs);
  smt_astt _rm = convert_rounding_mode(rm);
  return fp_api->mk_smt_fpbv_sub(_lhs, _rhs, _rm);
}

smt_astt smt_convt::mk_smt_fpbv_mul(expr2tc lhs, expr2tc rhs, expr2tc rm)
{
  smt_astt _lhs = convert_ast(lhs);
  smt_astt _rhs = convert_ast(rhs);
  smt_astt _rm = convert_rounding_mode(rm);
  return fp_api->mk_smt_fpbv_mul(_lhs, _rhs, _rm);
}

smt_astt smt_convt::mk_smt_fpbv_div(expr2tc lhs, expr2tc rhs, expr2tc rm)
{
  smt_astt _lhs = convert_ast(lhs);
  smt_astt _rhs = convert_ast(rhs);
  smt_astt _rm = convert_rounding_mode(rm);
  return fp_api->mk_smt_fpbv_div(_lhs, _rhs, _rm);
}
