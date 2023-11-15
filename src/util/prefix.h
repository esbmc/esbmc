#ifndef CPROVER_UTIL_PREFIX_H
#define CPROVER_UTIL_PREFIX_H

#include <string_view>

inline bool has_prefix(std::string_view s, std::string_view prefix)
{
  size_t n = prefix.size();
  return s.size() >= n && std::string_view(s.data(), n) == prefix;
}

#endif
