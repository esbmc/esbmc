#ifndef CPROVER_UTIL_PRETTY_H
#define CPROVER_UTIL_PRETTY_H

#include <string>

#include <irep2/irep2.h>
#include <irep2/irep2_expr.h>

inline std::string get_pretty_name(const std::string &name)
{
  return name.substr(name.find_last_of('@') + 1);
}

#endif // CPROVER_UTIL_PRETTY_H
