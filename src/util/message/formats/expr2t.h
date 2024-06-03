// LOOK AT <util/message/format.h> FOR MORE INFO ABOUT
// WHAT THIS FILE IS ABOUT!!!

#ifndef ESBMC_FORMATS_START_ASSERTION
#  error Do not include this header directly, use <util/message/format.h>
#endif

// expr2t Specialization
#include <irep2/irep2.h>
template <>
struct fmt::formatter<expr2t>
{
  // location does not support any specific format
  constexpr auto parse(format_parse_context &ctx)
  {
    auto it = ctx.begin(), end = ctx.end();
    if (it != end && *it != '}')
      throw format_error("invalid format");
    return it;
  }

  // This will teach fmt how to convert irep_idt into a str.
  template <typename FormatContext>
  auto format(const expr2t &p, FormatContext &ctx)
  {
    return format_to(ctx.out(), "{}", p.pretty(0));
  }
};
