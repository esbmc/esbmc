#include <assert.h>
#include <limits.h>

int main()
{
  _Bool c = nondet_bool();
  int x = nondet_int();
  int y = nondet_int();
  int z = nondet_int();

  // (c ? (c ? x : y) : z) -> (c ? x : z)
  assert((c ? (c ? x : y) : z) == (c ? x : z));

  // (c ? x : (c ? y : z)) → (c ? x : z)
  assert((c ? x : (c ? y : z)) == (c ? x : z));

  return 0;
}
