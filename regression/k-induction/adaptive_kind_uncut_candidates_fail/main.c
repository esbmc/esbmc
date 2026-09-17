/* Guesses mined for a loop the schema does not cut (it writes through a
 * pointer) were never emitted as claims, so no model could refute them, and
 * they were assumed in the inductive step: a wrong VERIFICATION SUCCESSFUL. */
#include <assert.h>
unsigned nondet_uint();
int main()
{
  unsigned m = nondet_uint();
  __ESBMC_assume(m <= 5);
  unsigned x = 0;
  unsigned *p = &x;
  unsigned j = 0;
  while (j < m)
  {
    j++;
    *p = j;
  }
  unsigned n = nondet_uint();
  unsigned i = 0;
  while (i < n)
    i++;
  assert(j != 5 || i < 50);
  return 0;
}
