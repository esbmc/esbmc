// LOOK AT <util/message/format.h> FOR MORE INFO ABOUT
// WHAT THIS FILE IS ABOUT!!!

#ifndef ESBMC_FORMATS_START_ASSERTION
#  error Do not include this header directly, use <util/message/format.h>
#endif

// side_effect_expr_function_callt Specialization
#include <util/std_code.h>
template <>
struct fmt::formatter<side_effect_expr_function_callt>
{
  // side_effect_expr_function_callt does not support any specific format
  constexpr auto parse(format_parse_context &ctx)
  {
    auto it = ctx.begin(), end = ctx.end();
    if (it != end && *it != '}')
      throw format_error("invalid format");
    return it;
  }

  // This will teach fmt how to convert side_effect_expr_function_callt into a str.
  template <typename FormatContext>
  auto format(const side_effect_expr_function_callt &p, FormatContext &ctx)
  {
    return format_to(ctx.out(), "{}", p.pretty(0));
  }
};
