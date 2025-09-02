/*******************************************************************
 Module: BigInt unit test

 Author: Rafael Sá Menezes

 Date: December 2019

 Fuzz Plan:
   - Constructors
   - Subset of operators
 \*******************************************************************/

#include <cctype>
#include <cassert>
#include <big-int/bigint.hh>
#include <vector>
#include <stddef.h>
#include <iostream>

// Checks whether the input contains digits only
bool is_valid_input(const char *Data, size_t DataSize)
{
  const char low_bound = '0';
  const char high_bound = '9';

  if (DataSize == 0)
    return false;
  if (Data[0] == '0')
    return false;
  // Last character must be a null terminator
  if (Data[DataSize - 1] != 0)
    return false;
  // Input could be negative
  if (
    !((Data[0] >= low_bound) && (Data[0] <= high_bound)) &&
    !(Data[0] == '-' && DataSize < 2))
    return false;
  for (size_t i = 1; i < DataSize - 1; ++i)
  {
    if (!((Data[i] >= low_bound) && (Data[i] <= high_bound)))
      return false;
  }
  return true;
}

BigInt test_construct_bigint(const char *Data, size_t DataSize)
{
  BigInt obj(Data, 10);

  std::vector<char> vec(obj.digits());
  const char *actual = obj.as_string(vec.data(), vec.size());

  for (size_t i = 0; i < DataSize - 1; ++i)
  {
    assert(Data[i] == actual[i]);
  }
  return obj;
}

extern "C" int LLVMFuzzerTestOneInput(const char *Data, size_t Size)
{
  if (!is_valid_input(Data, Size))
    return -1;
  BigInt bigint_a = test_construct_bigint(Data, Size);
  assert(bigint_a + bigint_a == bigint_a * 2);
  assert(bigint_a - bigint_a == 0);
  assert(bigint_a * 1 == bigint_a);
  assert(bigint_a / 1 == bigint_a);
  return 0;
}