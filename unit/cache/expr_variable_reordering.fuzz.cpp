/*******************************************************************
 Module: Expressions Variable Reordering unit tests

 Author: Rafael Sá Menezes

 Date: April 2020

 Fuzz Plan:

 1 - Generate input
 2 - Generate expr from input (unordered)
 3 - Generate expr from input (ordered)
 4 - Apply ordering in 2
 5 - Compare crc from 3 with 4
 \*******************************************************************/

#include <cctype>
#include <cassert>

extern "C" int LLVMFuzzerTestOneInput(const char *Data, size_t Size)
{
  test_construct_bigint(Data, Size);
  return 0;
}