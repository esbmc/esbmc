/*******************************************************************\

Module: Helper module for formatting text
Author: Rafael Sá Menezes, rafael.sa.menezes@outlook.com

\*******************************************************************/

#include <string>
/*******************************************************************\

Module: Specializations for formatting data from ESBMC

Author: Daniel Kroening, kroening@kroening.com

Maintainers:
- @2021: Rafael Sá Menezes, rafael.sa.menezes@outlook.com

\*******************************************************************/

#pragma once

#pragma GCC diagnostic push

// Clang has no support for nonnull-compare warning
#ifdef __GNUC__
#ifndef __clang__ // For some reason clang also defines GNUC :)
#pragma GCC diagnostic ignored "-Wnonnull-compare"
#endif
#endif

#pragma GCC diagnostic ignored "-Waddress"
#include <fmt/format.h>
#pragma GCC diagnostic pop
#include <big-int/bigint.hh>

// For more in-depth for how this specialization work look at
// https://fmt.dev/latest/api.html#format-api

// BigInt Specialization
template <>
struct fmt::formatter<BigInt>
{
  // Presentation format: 'd' - decimal (default). (Add on demand)
  char presentation = 'd'; // default

  // This will parse and look for the expected presentation mode
  constexpr auto parse(format_parse_context &ctx)
  {
    auto it = ctx.begin(), end = ctx.end();
    if(it != end && (*it == 'd'))
      presentation = *it++;
    if(it != end && *it != '}')
      throw format_error("invalid format");
    return it;
  }

  // This will teach fmt how to convert BigInt into a str.
  template <typename FormatContext>
  auto format(const BigInt &p, FormatContext &ctx)
  {
    int base;
    switch(presentation)
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
