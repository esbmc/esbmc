/* --multi-fail-fast skips the second claim once the first is violated. The
   forward condition then proves the program fully unwound, but a claim the
   run never solved is not thereby proved. */
#include <assert.h>

int main()
{
  unsigned n = nondet_uint();
  unsigned m = nondet_uint();
  assert(n != 7);
  assert(m + 1 > m || m == 0xffffffffu);
  return 0;
}
