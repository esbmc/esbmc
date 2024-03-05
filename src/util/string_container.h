#ifndef STRING_CONTAINER_H
#define STRING_CONTAINER_H

#include <cassert>
#include <list>
#include <unordered_map>
#include <string>
#include <string_view>
#include <vector>

typedef std::string_view string_ptrt;
typedef std::hash<string_ptrt> string_ptr_hash;

class string_containert
{
public:
  unsigned operator[](const char *s)
  {
    return get(s);
  }

  unsigned operator[](const std::string &s)
  {
    return get(s);
  }

  unsigned operator[](const string_ptrt &s)
  {
    return get(s);
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
    assert(no < string_vector.size());
    return string_vector[no]->c_str();
  }

  // the reference is guaranteed to be stable
  const std::string &get_string(size_t no) const
  {
    assert(no < string_vector.size());
    return *string_vector[no];
  }

protected:
  typedef std::unordered_map<string_ptrt, size_t, string_ptr_hash> hash_tablet;
  hash_tablet hash_table;

  unsigned get(string_ptrt s);

  typedef std::list<std::string> string_listt;
  string_listt string_list;

  typedef std::vector<std::string *> string_vectort;
  string_vectort string_vector;
};

inline string_containert &get_string_container()
{
  static string_containert ret;
  return ret;
}

#endif
