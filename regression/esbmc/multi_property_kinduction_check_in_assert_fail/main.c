/* The NULL-pointer check raised inside assert(*p == 2) fails at k = 1 and
   --multi-fail-fast skips the rest. The assertion is its own claim: the
   check's violation must not drop it, or no base case ever solves it. */
#include <assert.h>

int nondet_int();

int main()
{
  int x = 1;
  int *p = nondet_int() ? &x : 0;
  assert(*p == 2);
  return 0;
}
