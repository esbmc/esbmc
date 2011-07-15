/*******************************************************************\

Module: Container for C-Strings

Author: Daniel Kroening, kroening@kroening.com

\*******************************************************************/

#ifndef STRING_CONTAINER_H
#define STRING_CONTAINER_H

#include <assert.h>

#include <list>
#include <vector>

#include "hash_cont.h"
#include "string_hash.h"

struct string_ptrt
{
  const char *s;
  unsigned len;
  
  const char *c_str() const
  {
    return s;
  }
  
  explicit string_ptrt(const char *_s);

  explicit string_ptrt(const std::string &_s):s(_s.c_str()), len(_s.size())
  {
  }

  friend bool operator==(const string_ptrt a, const string_ptrt b);
};

bool operator==(const string_ptrt a, const string_ptrt b);

class string_ptr_hash
{
public:
  size_t operator()(const string_ptrt s) const { return hash_string(s.s); }
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

  // A class recording the state of the string table at a particular point in
  // time. Currently just wraps an integer describing the size of the string
  // vector.
  // The purpose of this class is so that we can explore the reachability tree
  // of some problem, and when we're done with a portion of it we can reset the
  // string table and eradicate strings generated during that exploration.
  //
  // Assumes that string identifiers have all been discarded by the time we
  // reset the string container state, so can't be used with, say, the schedule
  // option.
  class str_snapshot {
  public:
    str_snapshot() {
      idx = 0;
    }

    str_snapshot(unsigned long i) {
      idx = i;
    }

    unsigned long idx;
  };

  str_snapshot take_state_snapshot(void) {
    return str_snapshot(string_vector.size());
  }

  void restore_state_snapshot(str_snapshot &state);
  
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
