/*******************************************************************\

Module: STL Hash map / set

Author: Daniel Kroening, kroening@kroening.com

\*******************************************************************/

#ifndef CPROVER_HASH_CONT_H
#define CPROVER_HASH_CONT_H

// This used to provide the ability to pick which hash-map or hash-set
// implementation ESBMC / CBMC used. Seeing how we're now doing C++11, such
// structures are now part of the stl, so we don't need to think about that
// any more. Eventually the hash_set_cont and hash_map_cont symbols should
// be eliminated I guess.

#include <unordered_map>
#include <unordered_set>

// for new g++ libraries >= 3.2

// jmorse: boost.python (which is now everywhere) wants a key_comp() for all
// maps. Provide one for unordered_map. This is fine because it doesn't use
// the internal order of the map, only for it's own internal data structures.

template <typename Key, typename ...Args>
class esbmc_map_wrapper : public std::unordered_map<Key, Args...>
{
public:
  template <typename ...Args2>
  esbmc_map_wrapper(Args2 ...args)
  : std::unordered_map<Key, Args...>(args...) { }

  class key_compare {
  public:
    bool operator()(const Key &a, const Key &b) {
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

#endif
