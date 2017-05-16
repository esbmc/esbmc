/*******************************************************************\

Module: Container for C-Strings

Author: Daniel Kroening, kroening@kroening.com

\*******************************************************************/

#include <cassert>
#include <cstring>
#include <iostream>
#include <util/string_container.h>

string_containert string_container __attribute__((init_priority(101)));

string_ptrt::string_ptrt(const char *_s):s(_s), len(strlen(_s))
{
}

bool operator==(const string_ptrt a, const string_ptrt b)
{
  if(a.len!=b.len) return false;
  if(a.len==0) return true;
  return memcmp(a.s, b.s, a.len)==0;
}

unsigned string_containert::get(const char *s)
{
  string_ptrt string_ptr(s);

  hash_tablet::iterator it=hash_table.find(string_ptr);
  
  if(it!=hash_table.end())
    return it->second;

  unsigned r=hash_table.size();

  // these are stable
  string_list.push_back(std::string(s));
  string_ptrt result(string_list.back());

  hash_table[result]=r;
  
  // these are not
  string_vector.push_back(&string_list.back());

  return r;
}

unsigned string_containert::get(const std::string &s)
{
  string_ptrt string_ptr(s);

  hash_tablet::iterator it=hash_table.find(string_ptr);
  
  if(it!=hash_table.end())
    return it->second;

  unsigned r=hash_table.size();

  // these are stable
  string_list.push_back(s);
  string_ptrt result(string_list.back());

  hash_table[result]=r;
  
  // these are not
  string_vector.push_back(&string_list.back());

  return r;
}

// To avoid the static initialization order fiasco, it's important to have all
// the globals that interact with the string pool initialized in the same
// translation unit. This ensures that the string_container object is
// initialized before all of the attribute-name globals are. Somewhat miserable.

#include <expr.cpp>
#include <irep.cpp>
#include <type.cpp>
