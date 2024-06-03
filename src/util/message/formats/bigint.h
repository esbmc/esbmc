// LOOK AT <util/message/format.h> FOR MORE INFO ABOUT
// WHAT THIS FILE IS ABOUT!!!

#ifndef ESBMC_FORMATS_START_ASSERTION
#  error Do not include this header directly, use <util/message/format.h>
#endif

// BigInt Specialization
#include <big-int/bigint.hh>
template <>
struct fmt::formatter<BigInt>
{
  // Presentation format: 'd' - decimal (default). (Add on demand)
  char presentation = 'd'; // default

  // This will parse and look for the expected presentation mode
  constexpr auto parse(format_parse_context &ctx)
  {
    auto it = ctx.begin(), end = ctx.end();
    if (it != end && (*it == 'd'))
      presentation = *it++;
    if (it != end && *it != '}')
      throw format_error("invalid format");
    return it;
  }

  // This will teach fmt how to convert BigInt into a str.
  template <typename FormatContext>
  auto format(const BigInt &p, FormatContext &ctx)
  {
    int base;
    switch (presentation)
    {
    case 'd':
      base = 10;
      break;
    default:
      throw format_error("unsupported presentation for bigint");
    }
    char tmp[128];
    char *number;
    number = p.as_string(tmp, 128, base);
    return format_to(ctx.out(), "{}", number);
  }
};
