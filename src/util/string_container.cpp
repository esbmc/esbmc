#include <cassert>
#include <cstring>
#include <mutex>
#include <util/string_container.h>

unsigned string_containert::get(const std::string_view &s)
{  
  {
    std::shared_lock lock(string_container_mutex);
    hash_tablet::iterator it = hash_table.find(s);

    if (it != hash_table.end())
      return it->second;
  }

  std::unique_lock lock(string_container_mutex);
  //Recheck after acquiring sole lock
  hash_tablet::iterator it = hash_table.find(s);
  if (it != hash_table.end())
  {
    return it->second;
  }

  size_t r = hash_table.size();

  strings.emplace_back(s);
  std::string_view result(strings.back());

  hash_table[result] = r;

  return r;
}
