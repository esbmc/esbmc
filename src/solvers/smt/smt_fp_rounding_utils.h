#pragma once

#include "irep2/irep2_expr.h"
#include <util/ieee_float.h>

namespace smt_fp_rounding_utils
{
// Shared rounding-mode predicates used by both smt_conv.cpp and smt_fp_conv.cpp.
// Keeping these in one place avoids semantic drift between interval enclosure
// paths and IEEE754 post-processing paths.
inline bool is_nearest_rounding_mode(const expr2tc &rounding_mode)
{
  if (is_nil_expr(rounding_mode))
    return false;
  if (!is_constant_int2t(rounding_mode))
    return false;
  return to_constant_int2t(rounding_mode).value ==
         BigInt(ieee_floatt::ROUND_TO_EVEN);
}

inline bool is_round_to_plus_inf(const expr2tc &rounding_mode)
{
  if (is_nil_expr(rounding_mode))
    return false;
  if (!is_constant_int2t(rounding_mode))
    return false;
  return to_constant_int2t(rounding_mode).value ==
         BigInt(ieee_floatt::ROUND_TO_PLUS_INF);
}

inline bool is_round_to_minus_inf(const expr2tc &rounding_mode)
{
  if (is_nil_expr(rounding_mode))
    return false;
  if (!is_constant_int2t(rounding_mode))
    return false;
  return to_constant_int2t(rounding_mode).value ==
         BigInt(ieee_floatt::ROUND_TO_MINUS_INF);
}

inline bool is_round_to_zero(const expr2tc &rounding_mode)
{
  if (is_nil_expr(rounding_mode))
    return false;
  if (!is_constant_int2t(rounding_mode))
    return false;
  return to_constant_int2t(rounding_mode).value ==
         BigInt(ieee_floatt::ROUND_TO_ZERO);
}

inline bool is_round_to_away(const expr2tc &rounding_mode)
{
  if (is_nil_expr(rounding_mode))
    return false;
  if (!is_constant_int2t(rounding_mode))
    return false;
  return to_constant_int2t(rounding_mode).value ==
         BigInt(ieee_floatt::ROUND_TO_AWAY);
}
} // namespace smt_fp_rounding_utils
