#include <assert.h>

typedef unsigned int u32;
typedef unsigned long long u64;

unsigned int nondet_uint();

/* Assembling a 64-bit register value out of two 32-bit halves. The shift
   count is a plain signed int, but that says nothing about the range of the
   result: every value of hi leaves ((u64)hi << 32) representable in u64. */
int main()
{
  u32 hi = nondet_uint();
  u32 lo = nondet_uint();
  u64 v = ((u64)hi << 32) | lo;
  assert((u32)(v >> 32) == hi);
  return 0;
}
