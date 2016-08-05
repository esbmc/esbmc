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

  // Appease boost.python's error paths
  string_wrapper()
    : the_string(""), hash(hash_string("")) { }

  bool operator==(const string_wrapper &ref) const
  {
    if (hash != ref.hash)
      return false;

    if (the_string != ref.the_string)
      return false;

    return true;
  }

  bool operator<(const string_wrapper &ref) const
  {
    if (hash < ref.hash)
      return true;
    else if (hash > ref.hash)
      return false;

    // Hashes match,
    if (the_string < ref.the_string)
      return true;
    return false;
  }

  std::string the_string;
  size_t hash;
};

class string_wrap_hash {
public:
  size_t operator()(const string_wrapper &s) const { return s.hash; }
  size_t operator()(const string_wrapper &s1, const string_wrapper &s2) const
  { return s1 < s2; }
};

#endif
