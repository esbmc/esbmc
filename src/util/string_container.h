#ifndef STRING_CONTAINER_H
#define STRING_CONTAINER_H

#include <cassert>
#include <list>
#include <unordered_map>
#include <string>
#include <vector>

extern std::string zero;
struct string_ptrt
{
  const char *s;
  size_t len;

  const char *c_str() const
  {
    return s;
  }

  explicit string_ptrt(const char *_s);

  explicit string_ptrt(const std::string &_s) : s(_s.c_str()), len(_s.size())
  {
  }

  bool operator==(const string_ptrt &other) const;
};

class string_ptr_hash
{
public:
  size_t operator()(const string_ptrt s) const
  {
    return std::hash<std::string>{}(s.s);
  }
};

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

  string_containert()
  {
    // allocate empty string -- this gets index 0
    get("");
  }
  ~string_containert() = default;

  // the pointer is guaranteed to be stable
  const char *c_str(size_t no) const
  {
    if(no >= string_vector.size()) {
      return "0";
    }
    return string_vector[no]->c_str();
  }

  // the reference is guaranteed to be stable
  const std::string &get_string(size_t no) const
  {
    if(no >= string_vector.size()) {
      return zero;
    }
    return *string_vector[no];
  }

protected:
  typedef std::unordered_map<string_ptrt, size_t, string_ptr_hash> hash_tablet;
  hash_tablet hash_table;

  unsigned get(const char *s);
  unsigned get(const std::string &s);

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
