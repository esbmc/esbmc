#include <assert.h>

int main()
{
  // A | (A || B) -> A || B
  _Bool A, B;
  A = nondet_bool();
  B = nondet_bool();

  assert((A | (A || B)) == (A || B));
  assert(((A || B) | A) == (A || B));
  assert(((A | B) | A) == (A | B));
  return 0;
}
