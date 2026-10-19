/* github #8659: 0 - INT_MIN overflows, but the shared bit-vector encoding
 * only checked a > 0 for positive minus negative. */
#include <assert.h>
#include <limits.h>
int nondet_int();
int main(void)
{
  int a = nondet_int(), b = nondet_int(), r;
  long long diff = (long long)a - b;
  assert(
    __builtin_ssub_overflow(a, b, &r) == (diff > INT_MAX || diff < INT_MIN));
  assert(__builtin_ssub_overflow(0, INT_MIN, &r));
  assert(!__builtin_ssub_overflow(-1, INT_MIN, &r));
  return 0;
}
