/* github #8658: INT_MIN + INT_MIN wraps to 0 and was not reported as an
 * overflow by the shared bit-vector encoding. */
#include <assert.h>
#include <limits.h>
int nondet_int();
int main(void)
{
  int a = nondet_int(), b = nondet_int(), r;
  long long sum = (long long)a + b;
  assert(__builtin_sadd_overflow(a, b, &r) == (sum > INT_MAX || sum < INT_MIN));
  return 0;
}
