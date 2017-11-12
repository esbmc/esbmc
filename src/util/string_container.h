/*******************************************************************\

Module: Container for C-Strings

Author: Daniel Kroening, kroening@kroening.com

\*******************************************************************/

#ifndef STRING_CONTAINER_H
#define STRING_CONTAINER_H

#include <cassert>
#include <list>
#include <util/hash_cont.h>
#include <vector>

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
  typedef hash_map_cont<std::string, unsigned> hash_tablet;
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
