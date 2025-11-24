#ifndef STRING_CONTAINER_H
#define STRING_CONTAINER_H

#include <cassert>
#include <shared_mutex>
#include <unordered_map>
#include <string>
#include <deque>

class string_view_hash
{
public:
  using is_transparent = void;

  size_t operator()(std::string_view sv) const
  {
    return std::hash<std::string_view>{}(sv);
  }
};

class string_view_equal
{
public:
  using is_transparent = void;

  bool operator()(std::string_view a, std::string_view b) const
  {
    return a == b;
  }
};

class string_containert
{
public:
  template <typename T>
  unsigned operator[](const T &s)
  {
    return get(std::string_view(s));
  }

  string_containert()
  {
    // allocate empty string -- this gets index 0
    get("");
  }
  ~string_containert() = default;

  // the pointer is guaranteed to be stable
  const char *c_str(size_t no) const
  {
    std::shared_lock lock(string_container_mutex);
    assert(no < strings.size());
    return strings[no].c_str();
  }

  // the reference is guaranteed to be stable
  const std::string &get_string(size_t no) const
  {
    std::shared_lock lock(string_container_mutex);
    assert(no < strings.size());
    return strings[no];
  }

protected:
  typedef std::unordered_map<
    std::string_view,
    unsigned,
    string_view_hash,
    string_view_equal>
    hash_tablet;
  hash_tablet hash_table;

  mutable std::shared_mutex string_container_mutex;

  unsigned get(const std::string_view &s);
  std::deque<std::string> strings;
};

inline string_containert &get_string_container()
{
  static string_containert ret;
  return ret;
}

#endif
