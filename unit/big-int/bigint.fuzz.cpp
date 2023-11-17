/*******************************************************************
 Module: BigInt unit test

 Author: Rafael Sá Menezes

 Date: December 2019

 Fuzz Plan:
   - Constructors
 \*******************************************************************/

#include <cctype>
#include <cassert>
#include <big-int/bigint.hh>
#include <vector>
#include <stddef.h>

// Checks whether the input contains digits only
int is_valid_input(const char *Data, size_t DataSize)
{
  const char low_bound = '0';
  const char high_bound = '9';

  for (size_t i = 0; i < DataSize; ++i)
  {
    if (!((Data[i] >= low_bound) && (Data[i] <= high_bound)))
      return 0;
  }
  return 1;
}

void test_construct_bigint(const char *Data, size_t DataSize)
{
  if (DataSize == 0)
    return;
  if (Data[0] == '0')
    return;
  if (!is_valid_input(Data, DataSize))
    return;
  BigInt obj(Data, 10);

  std::vector<char> vec(obj.digits());
  const char *actual = obj.as_string(vec.data(), vec.size());

  for (size_t i = 0; i < DataSize; ++i)
  {
    assert(Data[i] == actual[i]);
  }
}

extern "C" int LLVMFuzzerTestOneInput(const char *Data, size_t Size)
{
  test_construct_bigint(Data, Size);
  return 0;
}