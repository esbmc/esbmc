#include <cassert>
#include <cstring>

#include <util/string_container.h>

unsigned string_containert::get(string_ptrt s)
{
  hash_tablet::iterator it = hash_table.find(s);
  if (it != hash_table.end())
    return it->second;

  // these are stable
  string_list.emplace_back(s);

  // these are not
  string_vector.push_back(&string_list.back());

  size_t r = hash_table.size();

  hash_table.emplace(string_list.back(), r);

  return r;
}

// To avoid the static initialization order fiasco, it's important to have all
// the globals that interact with the string pool initialized in the same
// translation unit. This ensures that the string_container object is
// initialized before all of the attribute-name globals are. Somewhat miserable.

#include <expr.cpp>
#include <irep.cpp>
#include <type.cpp>
