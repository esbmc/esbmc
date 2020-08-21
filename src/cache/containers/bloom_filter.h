/*******************************************************************\
 Module: Bloom Filter

 Author: Rafael Sá Menezes

 Date: April 2020

 A Bloom Filter is a probabilistic data-structure that
 checks if a element is *not* in a database or maybe
 is in it.

 It works by having an array of *m* bits used in conjunction
 of *k* hash keys. Checking it's membership through the assigned
 bits

\*******************************************************************/

#ifndef ESBMC_BLOOM_FILTER_H
#define ESBMC_BLOOM_FILTER_H

#include <set>
#include <bitset>
#include <iostream>
#include <functional>
#include <cassert>
#include <cmath>

template <typename T>
using bloom_filter_function = std::function<size_t(const T &)>;

template <typename T, size_t N, size_t K>
class bloom_filter
{
private:
  size_t num_of_elements = 0;

protected:
  const std::array<bloom_filter_function<T>, K> hashes;
  std::bitset<N> filter;

public:
  explicit bloom_filter(std::array<bloom_filter_function<T>, K> hashes)
    : hashes(hashes)
  {
  }
  double false_positive_ratio()
  {
    double p =
      pow(1 - exp(-hashes.size() * num_of_elements / N), hashes.size());
    return p;
  }
  double optimal_number_of_hashes()
  {
    assert(num_of_elements != 0);
    double k = N * log(2) / num_of_elements;
    return k;
  }

  void insert_element(const T &elem)
  {
    for(const auto &f : hashes)
    {
      filter.set(f(elem) % N, true);
    }
    ++num_of_elements;
  }
  bool test_element(const T &elem)
  {
    for(const auto &f : hashes)
    {
      if(!filter[f(elem) % N])
      {
        return false;
      }
    }
    return true;
  }

  void clear()
  {
    filter.reset();
  }
};
#endif //ESBMC_BLOOM_FILTER_H
