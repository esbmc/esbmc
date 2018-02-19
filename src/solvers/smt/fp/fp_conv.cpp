#include <solvers/smt/smt_conv.h>

fp_convt::fp_convt(smt_convt *_ctx) : ctx(_ctx)
{
}

smt_astt fp_convt::mk_smt_fpbv(const ieee_floatt &thereal)
{
  return ctx->mk_smt_bv(
    SMT_SORT_FAKE_FLOATBV, thereal.pack(), thereal.spec.width());
}

smt_sortt fp_convt::mk_fpbv_sort(const unsigned ew, const unsigned sw)
{
  std::cout << "Missing implementation of " << __FUNCTION__
            << " for the chosen solver\n";
  (void)ew;
  (void)sw;
  abort();
}

smt_sortt fp_convt::mk_fpbv_rm_sort()
{
  std::cout << "Missing implementation of " << __FUNCTION__
            << " for the chosen solver\n";
  abort();
}

smt_astt fp_convt::mk_smt_fpbv_nan(unsigned ew, unsigned sw)
{
  std::cout << "Missing implementation of " << __FUNCTION__
            << " for the chosen solver\n";
  (void)ew;
  (void)sw;
  abort();
}

smt_astt fp_convt::mk_smt_fpbv_inf(bool sgn, unsigned ew, unsigned sw)
{
  std::cout << "Missing implementation of " << __FUNCTION__
            << " for the chosen solver\n";
  (void)sgn;
  (void)ew;
  (void)sw;
  abort();
}

smt_astt fp_convt::mk_smt_fpbv_rm(ieee_floatt::rounding_modet rm)
{
  std::cout << "Missing implementation of " << __FUNCTION__
            << " for the chosen solver\n";
  (void)rm;
  abort();
}

smt_astt fp_convt::mk_smt_nearbyint_from_float(smt_astt from, smt_astt rm)
{
  std::cout << "Missing implementation of " << __FUNCTION__
            << " for the chosen solver\n";
  (void)from;
  (void)rm;
  abort();
}

smt_astt fp_convt::mk_smt_fpbv_sqrt(smt_astt rd, smt_astt rm)
{
  std::cout << "Missing implementation of " << __FUNCTION__
            << " for the chosen solver\n";
  (void)rd;
  (void)rm;
  abort();
}

smt_astt
fp_convt::mk_smt_fpbv_fma(smt_astt v1, smt_astt v2, smt_astt v3, smt_astt rm)
{
  std::cout << "Missing implementation of " << __FUNCTION__
            << " for the chosen solver\n";
  (void)v1;
  (void)v2;
  (void)v3;
  (void)rm;
  abort();
}

smt_astt fp_convt::mk_smt_typecast_from_fpbv_to_ubv(smt_astt from, smt_sortt to)
{
  std::cout << "Missing implementation of " << __FUNCTION__
            << " for the chosen solver\n";
  (void)from;
  (void)to;
  abort();
}

smt_astt fp_convt::mk_smt_typecast_from_fpbv_to_sbv(smt_astt from, smt_sortt to)
{
  std::cout << "Missing implementation of " << __FUNCTION__
            << " for the chosen solver\n";
  (void)from;
  (void)to;
  abort();
}

smt_astt fp_convt::mk_smt_typecast_from_fpbv_to_fpbv(
  smt_astt from,
  smt_sortt to,
  smt_astt rm)
{
  std::cout << "Missing implementation of " << __FUNCTION__
            << " for the chosen solver\n";
  (void)from;
  (void)to;
  (void)rm;
  abort();
}

smt_astt
fp_convt::mk_smt_typecast_ubv_to_fpbv(smt_astt from, smt_sortt to, smt_astt rm)
{
  std::cout << "Missing implementation of " << __FUNCTION__
            << " for the chosen solver\n";
  (void)from;
  (void)to;
  (void)rm;
  abort();
}

smt_astt
fp_convt::mk_smt_typecast_sbv_to_fpbv(smt_astt from, smt_sortt to, smt_astt rm)
{
  std::cout << "Missing implementation of " << __FUNCTION__
            << " for the chosen solver\n";
  (void)from;
  (void)to;
  (void)rm;
  abort();
}

ieee_floatt fp_convt::get_fpbv(smt_astt a)
{
  std::cout << "Missing implementation of " << __FUNCTION__
            << " for the chosen solver\n";
  (void)a;
  abort();
}

smt_astt fp_convt::mk_smt_fpbv_add(smt_astt lhs, smt_astt rhs, smt_astt rm)
{
  std::cout << "Missing implementation of " << __FUNCTION__
            << " for the chosen solver\n";
  (void)lhs;
  (void)rhs;
  (void)rm;
  abort();
}

smt_astt fp_convt::mk_smt_fpbv_sub(smt_astt lhs, smt_astt rhs, smt_astt rm)
{
  std::cout << "Missing implementation of " << __FUNCTION__
            << " for the chosen solver\n";
  (void)lhs;
  (void)rhs;
  (void)rm;
  abort();
}

smt_astt fp_convt::mk_smt_fpbv_mul(smt_astt lhs, smt_astt rhs, smt_astt rm)
{
  std::cout << "Missing implementation of " << __FUNCTION__
            << " for the chosen solver\n";
  (void)lhs;
  (void)rhs;
  (void)rm;
  abort();
}

smt_astt fp_convt::mk_smt_fpbv_div(smt_astt lhs, smt_astt rhs, smt_astt rm)
{
  std::cout << "Missing implementation of " << __FUNCTION__
            << " for the chosen solver\n";
  (void)lhs;
  (void)rhs;
  (void)rm;
  abort();
}

smt_astt fp_convt::mk_smt_fpbv_eq(smt_astt lhs, smt_astt rhs)
{
  std::cout << "Missing implementation of " << __FUNCTION__
            << " for the chosen solver\n";
  (void)lhs;
  (void)rhs;
  abort();
}

smt_astt fp_convt::mk_smt_fpbv_is_nan(smt_astt op)
{
  std::cout << "Missing implementation of " << __FUNCTION__
            << " for the chosen solver\n";
  (void)op;
  abort();
}

smt_astt fp_convt::mk_smt_fpbv_is_inf(smt_astt op)
{
  std::cout << "Missing implementation of " << __FUNCTION__
            << " for the chosen solver\n";
  (void)op;
  abort();
}

smt_astt fp_convt::mk_smt_fpbv_is_normal(smt_astt op)
{
  std::cout << "Missing implementation of " << __FUNCTION__
            << " for the chosen solver\n";
  (void)op;
  abort();
}

smt_astt fp_convt::mk_smt_fpbv_is_zero(smt_astt op)
{
  std::cout << "Missing implementation of " << __FUNCTION__
            << " for the chosen solver\n";
  (void)op;
  abort();
}

smt_astt fp_convt::mk_smt_fpbv_is_negative(smt_astt op)
{
  std::cout << "Missing implementation of " << __FUNCTION__
            << " for the chosen solver\n";
  (void)op;
  abort();
}

smt_astt fp_convt::mk_smt_fpbv_is_positive(smt_astt op)
{
  std::cout << "Missing implementation of " << __FUNCTION__
            << " for the chosen solver\n";
  (void)op;
  abort();
}

smt_astt fp_convt::mk_smt_fpbv_abs(smt_astt op)
{
  std::cout << "Missing implementation of " << __FUNCTION__
            << " for the chosen solver\n";
  (void)op;
  abort();
}
