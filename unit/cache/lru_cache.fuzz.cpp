/*******************************************************************\
 Module: LRU fuzz tests

 Author: Rafael Sá Menezes

 Date: April 2020

 Fuzz Plan:

 1 - Generate input
 2 - Check size
 3 - Check overlaps
\*******************************************************************/

#include <cache/containers/lru_cache.h>
#include <set>
//#include <vector>
using fuzzing_type = int;
using fuzz_lru = lru_cache<fuzzing_type>;

/*
bool is_input_unique(const fuzzing_type *Data, const size_t Size)
{
  std::vector<int> v;
  for(size_t i = 0; i < Size; i++)
    v.push_back(Data[i]);

  std::sort(v.begin(), v.end());
  auto unique = std::unique(v.begin(), v.end());
  v.erase(unique, v.end());
  return Size == v.size();
}
*/
bool is_valid_input(const fuzzing_type *Data, const size_t Size)
{
  if(Size < 10)
    return false;
  return true;
}

void test_insertion(const fuzzing_type *Data, size_t Size)
{
  const size_t cache_length = 20;
  fuzz_lru lru(cache_length);
  for(size_t i = 0; i < Size; i++)
  {
    lru.insert(Data[i]);
    assert(lru.exists(Data[i]));
  }

  if(Size > cache_length)
  {
    assert(!lru.exists(Data[0]));
  }
}

extern "C" int LLVMFuzzerTestOneInput(const fuzzing_type *Data, size_t Size)
{
  if(!is_valid_input(Data, Size))
    return 0;
  test_insertion(Data, Size);
  return 0;
}