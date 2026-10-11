#pragma once

#include <util/config/config.h>
#include <cerrno>
#include <cstdlib>
#include <optional>
#include <stdexcept>
#include <string>

struct LdCycleTime
{
  long long min_ms;
  long long max_ms;
};

// --ld-cycle-time MIN:MAX: every scan lasts between MIN and MAX milliseconds,
// timers count in milliseconds (docs/safe-ld-sos-semantics.md, timing mode).
// Empty when the option is unset; throws std::invalid_argument when malformed.
inline std::optional<LdCycleTime> ld_cycle_time()
{
  const std::string s = config.options.get_option("ld-cycle-time");
  if (s.empty())
    return std::nullopt;

  auto parse = [](const std::string &part) -> long long {
    errno = 0;
    char *end = nullptr;
    const long long v = std::strtoll(part.c_str(), &end, 10);
    if (part.empty() || *end != '\0' || errno == ERANGE)
      throw std::invalid_argument("");
    return v;
  };

  try
  {
    const size_t colon = s.find(':');
    if (colon == std::string::npos)
      throw std::invalid_argument("");
    const LdCycleTime t{parse(s.substr(0, colon)), parse(s.substr(colon + 1))};
    if (t.min_ms < 1 || t.max_ms < t.min_ms)
      throw std::invalid_argument("");
    return t;
  }
  catch (const std::invalid_argument &)
  {
    throw std::invalid_argument(
      "invalid --ld-cycle-time '" + s +
      "': expected MIN:MAX in milliseconds with 1 <= MIN <= MAX");
  }
}
