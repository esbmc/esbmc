#ifndef CPROVER_NUMBERING_H
#define CPROVER_NUMBERING_H

#include <cassert>
#include <cstddef>
#include <map>
#include <unordered_map>
#include <vector>
#include <cstddef>

template <typename T>
class numbering
{
public:
  numbering()
  {
    next_obj_num = 0;
  }

  unsigned number(const T &a)
  {
    unsigned int num = next_obj_num;
    std::pair<typename numberst::const_iterator, bool> result =
      numbers.insert(std::pair<T, unsigned>(a, num));

    if (result.second) // inserted?
    {
      next_obj_num++;
      vec[num] = a;
      assert(vec.size() == numbers.size());
    }

    return (result.first)->second;
  }

  bool get_number(const T &a, unsigned &n) const
  {
    typename numberst::const_iterator it = numbers.find(a);

    if (it == numbers.end())
      return true;

    n = it->second;
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

  void erase(unsigned int num)
  {
    // Precondition: this number actually exists.
    assert(vec.find(num) != vec.end());

    const T &ref = vec[num];
    numbers.erase(ref);
    vec.erase(num);
    return;
  }

protected:
  typedef std::map<T, unsigned> numberst;
  typedef std::unordered_map<unsigned, T, std::hash<unsigned>> vectort;
  numberst numbers;
  vectort vec;
  unsigned int next_obj_num;
};

template <typename T, class hash_fkt>
class hash_numbering
{
public:
  hash_numbering()
  {
    next_obj_num = 0;
  }

  unsigned number(const T &a)
  {
    unsigned int num = next_obj_num++;
    std::pair<typename numberst::const_iterator, bool> result =
      numbers.insert(std::pair<T, unsigned>(a, num));

    if (result.second) // inserted?
    {
      vec[num] = a;
      assert(vec.size() == numbers.size());
    }

    return (result.first)->second;
  }

  bool get_number(const T &a, unsigned &n) const
  {
    typename numberst::const_iterator it = numbers.find(a);

    if (it == numbers.end())
      return true;

    n = it->second;
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

  const T &operator[](unsigned int i) const
  {
    return vec[i];
  }

  T &operator[](unsigned int i)
  {
    return vec[i];
  }

  void erase(unsigned int num)
  {
    assert(vec.find(num) != vec.end());
    const T &ref = vec[num];
    numbers.erase(ref);
    vec.erase(num);
  }

protected:
  typedef std::unordered_map<T, unsigned, hash_fkt> numberst;
  typedef std::unordered_map<unsigned, T, std::hash<unsigned>> vectort;
  numberst numbers;
  vectort vec;
  unsigned int next_obj_num;
};

#endif
