#include <solvers/smt/smt_conv.h>
#include <solvers/smt/fp/float_bv.h>

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
  return convert_ast(
    float_bvt::to_unsigned_integer(from, width, float_bvt::get_spec(from)));
}

smt_astt
smt_convt::mk_smt_typecast_from_fpbv_to_sbv(expr2tc from, std::size_t width)
{
  return convert_ast(
    float_bvt::to_signed_integer(from, width, float_bvt::get_spec(from)));
}

smt_astt smt_convt::mk_smt_typecast_from_fpbv_to_fpbv(
  expr2tc from,
  type2tc to,
  expr2tc rm)
{
  return convert_ast(float_bvt::conversion(
    from,
    rm,
    float_bvt::get_spec(from),
    ieee_float_spect(to_floatbv_type(to))));
}

smt_astt
smt_convt::mk_smt_typecast_ubv_to_fpbv(expr2tc from, type2tc to, expr2tc rm)
{
  return convert_ast(float_bvt::from_unsigned_integer(
    from, rm, ieee_float_spect(to_floatbv_type(to))));
}

smt_astt
smt_convt::mk_smt_typecast_sbv_to_fpbv(expr2tc from, type2tc to, expr2tc rm)
{
  return convert_ast(float_bvt::from_signed_integer(
    from, rm, ieee_float_spect(to_floatbv_type(to))));
}

smt_astt smt_convt::mk_smt_fpbv_add(expr2tc lhs, expr2tc rhs, expr2tc rm)
{
  return convert_ast(float_bvt::add_sub(
    false, lhs, rhs, rm, ieee_float_spect(to_floatbv_type(lhs->type))));
}

smt_astt smt_convt::mk_smt_fpbv_sub(expr2tc lhs, expr2tc rhs, expr2tc rm)
{
  return convert_ast(float_bvt::add_sub(
    true, lhs, rhs, rm, ieee_float_spect(to_floatbv_type(lhs->type))));
}

smt_astt smt_convt::mk_smt_fpbv_mul(expr2tc lhs, expr2tc rhs, expr2tc rm)
{
  return convert_ast(
    float_bvt::mul(lhs, rhs, rm, ieee_float_spect(to_floatbv_type(lhs->type))));
}

smt_astt smt_convt::mk_smt_fpbv_div(expr2tc lhs, expr2tc rhs, expr2tc rm)
{
  return convert_ast(
    float_bvt::div(lhs, rhs, rm, ieee_float_spect(to_floatbv_type(lhs->type))));
}
