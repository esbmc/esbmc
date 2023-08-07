/*******************************************************************
 Module: Builtin Libs Model unit test

 Author: Rafael Sá Menezes

 Date: December 2019

 Fuzz Plan:
   - Functions
 \*******************************************************************/

#include <stdlib.h>
#include <cassert>
#include <stddef.h>

void __ESBMC_atomic_begin()
{
}
void __ESBMC_atomic_end()
{
}

bool is_valid_input(const int *, size_t Size)
{
  return Size >= 2 && Size < 3;
}

void test_sync_fetch_add(int initial, int value)
{
  int actual = initial;
  int num = value;
  int expected = initial + value;
  int fetch = __sync_fetch_and_add(&actual, num);
  assert(expected == actual);
  assert(fetch == initial);
}

extern "C" int LLVMFuzzerTestOneInput(const int *Data, size_t Size)
{
  if(!is_valid_input(Data, Size))
    return 0;
  test_sync_fetch_add(Data[0], Data[1]);
  return 0;
}