#ifndef CPROVER_UTIL_PREFIX_H
#define CPROVER_UTIL_PREFIX_H

#include <string_view>

inline bool has_prefix(std::string_view s, std::string_view prefix)
{
  size_t n = prefix.size();
  return s.size() >= n && std::string_view(s.data(), n) == prefix;
}

inline bool has_suffix(std::string_view s, std::string_view suffix)
{
  size_t n = suffix.size();
  size_t m = s.size();
  return m >= n && std::string_view(s.data() + (m - n), n) == suffix;
}

#endif
