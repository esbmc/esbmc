/*******************************************************************\
 Module: LRU Cache
 
 Author: Rafael Sá Menezes

 Date: March 2020

 This cache is used to keep track of which data was accessed recently and keep
 removing old entries.
\*******************************************************************/

#ifndef LRU_CACHE
#define LRU_CACHE

#include <list>
#include <algorithm>
#include <unordered_map>
#include <stdexcept>

template <typename K>
class lru_cache
{
protected:
  // std::list is a double ended queue
  std::list<K> cache;
  size_t max_capacity = 0;

public:
  /**
   * Constructs LRU Cache
   * @param length max size of cache
   */
  explicit lru_cache(size_t length) : max_capacity(length)
  {
  }
  virtual ~lru_cache()
  {
    cache.clear();
  }
  /**
   * Insert element into cache
   * @param key key of the element
   */
  void insert(const K &key)
  {
    if(exists(key))
    {
      return;
    }
    while(cache.size() >= max_capacity)
    {
      cache.pop_back();
    }
    cache.emplace_front(key);
  }

  /**
   * Checks whether the cache contains the element
   * @param key index of the element
   * @return boolean representing if the element is on the cache
   */
  bool exists(const K &key) const
  {
    return std::find(cache.begin(), cache.end(), key) != cache.end();
  }

  /**
   * Get current size of the cache
   * @return
   */
  size_t size()
  {
    return cache.size();
  }

  /**
   * Get max capacity of the cache
   * @return
   */
  constexpr size_t max_size()
  {
    return max_capacity;
  }
};
#endif
