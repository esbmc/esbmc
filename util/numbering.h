/*******************************************************************\

Module:

Author: Daniel Kroening, kroening@kroening.com

\*******************************************************************/

#ifndef CPROVER_NUMBERING_H
#define CPROVER_NUMBERING_H

#include <assert.h>
#include <unistd.h>

#include <map>
#include <vector>

#include "hash_cont.h"

template <typename T>
class numbering
{
public:
  unsigned number(const T &a)
  {
    std::pair<typename numberst::const_iterator, bool> result=
      numbers.insert(
      std::pair<T, unsigned>
      (a, numbers.size()));

    if(result.second) // inserted?
    {
      vec.push_back(a);
      assert(vec.size()==numbers.size());
    }
    
    return (result.first)->second;
  }
  
  bool get_number(const T &a, unsigned &n) const
  {
    typename numberst::const_iterator it=numbers.find(a);

    if(it==numbers.end())
      return true;
      
    n=it->second;
    return false;
  }

  void clear()
  {
    vec.clear();
    numbers.clear();
  }

  size_t size()
  {
    return vec.size();
  }

protected:
  typedef std::map<T, unsigned> numberst;
  typedef std::vector<T> vectort;
  numberst numbers;  
  vectort vec;
};

template <typename T, class hash_fkt>
class hash_numbering
{
public:
  unsigned number(const T &a)
  {
    std::pair<typename numberst::const_iterator, bool> result=
      numbers.insert(
      std::pair<T, unsigned>
      (a, numbers.size()));

    if(result.second) // inserted?
    {
      vec.push_back(a);
      assert(vec.size()==numbers.size());
    }
    
    return (result.first)->second;
  }
  
  bool get_number(const T &a, unsigned &n) const
  {
    typename numberst::const_iterator it=numbers.find(a);

    if(it==numbers.end())
      return true;
      
    n=it->second;
    return false;
  }

  void clear()
  {
    vec.clear();
    numbers.clear();
  }

  size_t size()
  {
    return vec.size();
  }

  const T &operator[](unsigned int i)
  {
    return vec[i];
  }

protected:
  typedef hash_map_cont<T, unsigned, hash_fkt> numberst;
  typedef std::vector<T> vectort;
  numberst numbers;  
  vectort vec;
};

#endif
