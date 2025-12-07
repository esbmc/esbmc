#include <assert.h>
#include <limits.h>

int main()
{
  _Bool c = nondet_bool();
  _Bool d = nondet_bool();
  int x = nondet_int();
  int y = nondet_int();
  int z = nondet_int();

  // (c ? (c ? x : y) : z) -> (c ? x : z)
  assert((c ? (c ? x : y) : z) == (c ? x : z));

  // (c ? x : (c ? y : z)) → (c ? x : z)
  assert((c ? x : (c ? y : z)) == (c ? x : z));

  // (c ? x : x) → x
  assert((c ? x : x) == x);

  // (!c ? x : y) == (c ? y : x)
  //assert((!c ? x : y) == (c ? y : x));

  const int K = 2;

  // (c ? K : K == K)
  assert((c ? K : K) == K);

  // (c ? x : (!c ? y : z)) == (c ? x : y)
  assert((c ? x : (!c ? y : z)) == (c ? x : y));

  // (!c ? x : (c ? y : z)) == (c ? y : x)
  assert((!c ? x : (c ? y : z)) == (c ? y : x));

  return 0;
}
