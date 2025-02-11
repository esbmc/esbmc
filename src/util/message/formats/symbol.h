// LOOK AT <util/message/format.h> FOR MORE INFO ABOUT
// WHAT THIS FILE IS ABOUT!!!

#ifndef ESBMC_FORMATS_START_ASSERTION
#  error Do not include this header directly, use <util/message/format.h>
#endif

// symbol Specialization
#include <util/symbol.h>
template <>
struct fmt::formatter<symbolt>
{
  // symbol does not support any specific format
  constexpr auto parse(format_parse_context &ctx)
  {
    auto it = ctx.begin(), end = ctx.end();
    if (it != end && *it != '}')
      throw format_error("invalid format");
    return it;
  }

  // This will teach fmt how to convert irep_idt into a str.
  template <typename FormatContext>
  auto format(const symbolt &p, FormatContext &ctx)
  {
    std::ostringstream oss;
    p.show(oss);
    return format_to(ctx.out(), "{}", oss.str());
  }
};
