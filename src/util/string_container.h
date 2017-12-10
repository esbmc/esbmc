/*******************************************************************\

Module: Container for C-Strings

Author: Daniel Kroening, kroening@kroening.com

\*******************************************************************/

#ifndef STRING_CONTAINER_H
#define STRING_CONTAINER_H

#include <cassert>
#include <list>
#include <util/hash_cont.h>
#include <util/string_hash.h>
#include <vector>

struct string_ptrt
{
  const char *s;
  unsigned len;

  const char *c_str() const
  {
    return s;
  }

  explicit string_ptrt(const char *_s);

  explicit string_ptrt(const std::string &_s) : s(_s.c_str()), len(_s.size())
  {
  }

  friend bool operator==(const string_ptrt a, const string_ptrt b);
};

bool operator==(const string_ptrt a, const string_ptrt b);

class string_ptr_hash hash_map_hasher_superclass(
  std::string){public: size_t operator()(const string_ptrt s)
                 const {return hash_string(s.s);
}
bool operator()(const string_ptrt &s1, const string_ptrt &s2) const
{
  return hash_string(s1.s) < hash_string(s2.s);
}
}
;

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

  const char *c_str(unsigned no) const
  {
    assert(no < string_vector.size());
    return string_vector[no]->c_str();
  }

  const std::string &get_string(unsigned no) const
  {
    assert(no < string_vector.size());
    return *string_vector[no];
  }

protected:
  typedef hash_map_cont<string_ptrt, unsigned, string_ptr_hash> hash_tablet;
  hash_tablet hash_table;

  unsigned get(const char *s);
  unsigned get(const std::string &s);

  typedef std::list<std::string> string_listt;
  string_listt string_list;

  typedef std::vector<std::string *> string_vectort;
  string_vectort string_vector;
};

extern string_containert string_container;

#endif
