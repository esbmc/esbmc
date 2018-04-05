/*******************************************************************\

Module: STL Hash map / set

Author: Daniel Kroening, kroening@kroening.com

\*******************************************************************/

#ifndef CPROVER_HASH_CONT_H
#define CPROVER_HASH_CONT_H

// you need to pick one of the following three options

// Use STDEXT hashes on MSVC, gnu otherwise.

// #define NO_HASH_CONT
#ifdef _MSC_VER
#define STDEXT_HASH_CONT
#else
#define GNU_HASH_CONT
#endif
// #define TR1_HASH_CONT

#ifdef NO_HASH_CONT

#include <map>
#include <set>

template <class T1, class T2, class T3>
typedef std::map<T1, T2> hash_map_cont;

template <class T1, class T2>
typedef std::set<T1> hash_set_cont;

template <class T1, class T2>
typedef std::multiset<T1> hash_multiset_cont;

#define hash_map_hasher_superclass(type)

#else
#ifdef STDEXT_HASH_CONT
#include <hash_map>
#include <hash_set>

// for Visual Studio >= 2003

#define hash_map_cont stdext::hash_map
#define hash_set_cont stdext::hash_set
#define hash_multiset_cont stdext::hash_multiset
#define hash_map_hasher_superclass(type) : public stdext::hash_compare<type>

#else

#ifdef GNU_HASH_CONT

#include <unordered_map>
#include <unordered_set>

// for new g++ libraries >= 3.2

// jmorse: boost.python (which is now everywhere) wants a key_comp() for all
// maps. Provide one for unordered_map. This is fine because it doesn't use
// the internal order of the map, only for it's own internal data structures.

template <typename Key, typename... Args>
class esbmc_map_wrapper : public std::unordered_map<Key, Args...>
{
public:
  template <typename... Args2>
  esbmc_map_wrapper(Args2... args) : std::unordered_map<Key, Args...>(args...)
  {
  }

  class key_compare
  {
  public:
    bool operator()(const Key &a, const Key &b)
    {
      return a < b;
    }
  };
  key_compare key_comp()
  {
    return key_compare();
  }
};

#define hash_map_cont esbmc_map_wrapper
#define hash_set_cont std::unordered_set
#define hash_multiset_cont __gnu_cxx::hash_multiset
#define hash_map_hasher_superclass(type)

#else

#ifdef TR1_HASH_CONT

#ifdef _MSC_VER
#include <unordered_set>
#include <unordered_map>

#else
#include <tr1/unordered_set>
#include <tr1/unordered_map>

#endif

#define hash_map_cont std::tr1::unordered_map
#define hash_set_cont std::tr1::unordered_set
#define hash_multiset_cont std::tr1::unordered_multiset
#define hash_map_hasher_superclass(type)

#else

#error Please select hash container option

#endif

#endif

#endif
#endif

#endif
