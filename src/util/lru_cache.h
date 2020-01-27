/*******************************************************************\
 
Module: LRU Cache
 
Author: Rafael Menezes, rafael.sa.menezes40@gmail.com
 
\*******************************************************************/

#ifndef LRU_CACHE
#define LRU_CACHE

#include <list>
#include <algorithm>
#include <unordered_map>

template <typename K, typename V>
class lru_cache
{
protected:
  typedef typename std::pair<K, V> key_value;
  typedef typename std::list<key_value>::iterator key_value_iterator;

  // std::list is a double ended queue
  std::list<key_value> cache;
  // a list has O(N) worst case, so it should be
  // faster to use a Hash for the index
  std::unordered_map<K, key_value_iterator> cache_index;
  size_t max_capacity = 0;

public:
  lru_cache(size_t length) : max_capacity(length)
  {
  }

  void insert(const K &key, const V &value)
  {
  }

  V &get(const K &key)
  {
    return 0;
  }

  bool exists(const K &key)
  {
    return false;
  }

  size_t size()
  {
    return 0;
  }

  const size_t max_size()
  {
    return max_capacity;
  }
};
#endif