#include <assert.h>
#include <stddef.h>
#include <stdio.h>

int main()
{
  // %ld of 42L = "42" (2 chars)
  int r1 = printf("%ld", (long)42);
  assert(r1 == 2);

  // %zu of (size_t)0 = "0" (1 char)
  int r2 = printf("%zu", (size_t)0);
  assert(r2 == 1);

  // %llX of 0xFFFFFFFFFFFFFFFFULL = "FFFFFFFFFFFFFFFF" (16 chars)
  int r3 = printf("%llX", (unsigned long long)0xFFFFFFFFFFFFFFFFULL);
  assert(r3 == 16);

  // %hhd truncates to char — -1 prints as "-1" (2 chars)
  int r4 = printf("%hhd", (signed char)-1);
  assert(r4 == 2);
}
