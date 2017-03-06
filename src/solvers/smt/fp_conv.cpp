/*
 * fp_conv.cpp
 *
 *  Created on: Mar 3, 2017
 *      Author: mramalho
 */

#include "fp_conv.h"

fp_convt::fp_convt(smt_convt *_ctx) : ctx(_ctx)
{
}

smt_astt fp_convt::mk_smt_fpbv(const ieee_floatt& thereal)
{
  std::cerr << __PRETTY_FUNCTION__ <<
    " not implemented for the chosen solver" << std::endl;
  (void) thereal;
  abort();
}

smt_astt fp_convt::mk_smt_fpbv_nan(unsigned ew, unsigned sw)
{
  std::cerr << __PRETTY_FUNCTION__ <<
    " not implemented for the chosen solver" << std::endl;
  (void) ew;
  (void) sw;
  abort();
}

smt_astt fp_convt::mk_smt_fpbv_inf(bool sgn, unsigned ew, unsigned sw)
{
  std::cerr << __PRETTY_FUNCTION__ <<
    " not implemented for the chosen solver" << std::endl;
  (void) sgn;
  (void) ew;
  (void) sw;
  abort();
}

smt_astt fp_convt::mk_smt_fpbv_rm(ieee_floatt::rounding_modet rm)
{
  std::cerr << __PRETTY_FUNCTION__ <<
    " not implemented for the chosen solver" << std::endl;
  (void) rm;
  abort();
}

smt_astt fp_convt::mk_smt_typecast_from_fpbv(const typecast2t& cast)
{
  std::cerr << __PRETTY_FUNCTION__ <<
    " not implemented for the chosen solver" << std::endl;
  (void) cast;
  abort();
}

smt_astt fp_convt::mk_smt_typecast_to_fpbv(const typecast2t& cast)
{
  std::cerr << __PRETTY_FUNCTION__ <<
    " not implemented for the chosen solver" << std::endl;
  (void) cast;
  abort();
}

smt_astt fp_convt::mk_smt_nearbyint_from_float(const nearbyint2t& expr)
{
  std::cerr << __PRETTY_FUNCTION__ <<
    " not implemented for the chosen solver" << std::endl;
  (void) expr;
  abort();
}

smt_astt fp_convt::mk_smt_fpbv_arith_ops(const expr2tc& expr)
{
  std::cerr << __PRETTY_FUNCTION__ <<
    " not implemented for the chosen solver" << std::endl;
  (void) expr;
  abort();
}

smt_astt fp_convt::mk_smt_fpbv_fma(const expr2tc& expr)
{
  std::cerr << __PRETTY_FUNCTION__ <<
    " not implemented for the chosen solver" << std::endl;
  (void) expr;
  abort();
}

smt_sortt fp_convt::mk_fpbv_sort(const unsigned ew, const unsigned sw)
{
  std::cerr << __PRETTY_FUNCTION__ <<
    " not implemented for the chosen solver" << std::endl;
  (void) ew;
  (void) sw;
  abort();
}

expr2tc fp_convt::get_fpbv(smt_astt a)
{
  std::cerr << __PRETTY_FUNCTION__ <<
    " not implemented for the chosen solver" << std::endl;
  (void) a;
  abort();
}
