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

template<class T1, class T2, class T3>
typedef std::map<T1, T2> hash_map_cont;

template<class T1, class T2>
typedef std::set<T1> hash_set_cont;

template<class T1, class T2>
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

#include <unordered_set>
#include <unordered_map>

// for new g++ libraries >= 3.2

#define hash_map_cont std::unordered_map
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
