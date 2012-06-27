/*******************************************************************\

Module: string hashing

Author: Daniel Kroening, kroening@kroening.com

\*******************************************************************/

#ifndef CPROVER_STRING_HASH_H
#define CPROVER_STRING_HASH_H

#include <string>

size_t hash_string(const std::string &s);
size_t hash_string(const char *s);

struct string_hash
{
  size_t operator()(const std::string &s) const { return hash_string(s); }
};

class string_wrapper {
public:
  string_wrapper(const std::string &str)
    : the_string(str), hash(hash_string(str)) { }

  std::string the_string;
  size_t hash;
};

class string_wrap_hash {
public:
  size_t operator()(const string_wrapper &s) const { return s.hash; }
  size_t operator()(const string_wrapper &s1, const string_wrapper &s2) const
  {
    if (s1.hash < s2.hash)
      return true;
    else if (s1.hash > s2.hash)
      return false;

    // Hashes match,
    if (s1.the_string < s2.the_string)
      return true;
    return false;
  }
};

#endif
