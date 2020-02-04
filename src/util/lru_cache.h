/*******************************************************************\
 
Module: LRU Cache
 
Author: Rafael Menezes, rafael.sa.menezes40@gmail.com
 
\*******************************************************************/

#ifndef LRU_CACHE
#define LRU_CACHE

#include <list>
#include <algorithm>
#include <unordered_map>
#include <stdexcept>

//! Least Recently Used Cache */
/*!
  This cache is used to keep track of which data was accessed recently and keep
  removing old entries.
*/
template <typename K, typename V>
class lru_cache
{
protected:
  typedef typename std::pair<K, V> KV;
  typedef typename std::list<KV>::iterator KV_iterator;
  // std::list is a double ended queue
  std::list<KV> cache;
  // a list has O(N) worst case, so it should be
  // faster to use a Hash for indexation O(1)
  std::unordered_map<K, KV_iterator> cache_index;
  size_t max_capacity = 0;

public:
  //! Constructs LRU Cache
  //! \param length max size of cache
  lru_cache(size_t length) : max_capacity(length)
  {
  }

  //! Insert element into cache
  //! \param key key of the element
  //! \param value value of the element
  void insert(const K &key, const V &value)
  {
    if(exists(key))
    {
      KV_iterator &it = cache_index[key];
      it->second = value;
      return;
    }
    cache.push_front(KV(key, value));
    cache_index[key] = cache.begin();
    if(cache.size() > max_size())
    {
      KV &item = cache.back();
      cache_index.erase(item.first);
      cache.pop_back();
    }
  }

  //! Try to return the value from key, throw exception otherwise
  //! \param key index to the value
  //! \return reference to the element
  V &get(const K &key)
  {
    if(!exists(key))
      throw std::range_error("Item is not on cache!");

    KV_iterator &it = cache_index[key];
    KV &item = *it;
    return item.second;
  }

  //! Checks whether the cache contains the element
  //! \param key index of the element
  //! \return boolean representing if the element is on the cache
  bool exists(const K &key)
  {
    return cache_index.find(key) != cache_index.end();
  }

  //! Get current size of the cache
  //! \return
  size_t size()
  {
    return cache.size();
  }

  //! Get max capacity of the cache
  //! \return
  const size_t max_size()
  {
    return max_capacity;
  }

  const size_t max_size()
  {
    return max_capacity;
  }
};
#endif
