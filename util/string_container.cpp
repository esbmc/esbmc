/*******************************************************************\

Module: Container for C-Strings

Author: Daniel Kroening, kroening@kroening.com

\*******************************************************************/

#include <iostream>

#include <assert.h>
#include <string.h>

#include "string_container.h"

string_containert string_container;

/*******************************************************************\

Function: string_ptrt::string_ptrt

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

string_ptrt::string_ptrt(const char *_s):s(_s), len(strlen(_s))
{
}

/*******************************************************************\

Function: operator==

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

bool operator==(const string_ptrt a, const string_ptrt b)
{
  if(a.len!=b.len) return false;
  if(a.len==0) return true;
  return memcmp(a.s, b.s, a.len)==0;
}

/*******************************************************************\

Function: string_containert::get

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

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

/*******************************************************************\

Function: string_containert::get

  Inputs:

 Outputs:

 Purpose:

\*******************************************************************/

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

void string_containert::restore_state_snapshot(str_snapshot &state)
{
  unsigned long i, to_free;

  assert(state.idx != 0);

  // Iterate over all strings that have been allocated since the snapshot and
  // free them from string_containert's hash table.
  for (i = state.idx, to_free = 0; i < string_vector.size(); i++, to_free++) {
    std::string *foo = string_vector[i];
    string_ptrt ptr(*foo);

    hash_table.erase(ptr);
  }

  // Resize string vector to remove all string ptrs too
  string_vector.erase(string_vector.begin() + state.idx, string_vector.end());

  // Finally, pop all string records themselves from the string list.
  // It might be more efficient to use the erase method.
  for (; to_free > 0; to_free--)
    string_list.pop_back();

  return;
}
