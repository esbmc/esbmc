/* #7516: a base case that only refutes the loop-invariant abstraction must
   not stop the search for a real bug at a larger k. */
#include <assert.h>
int nondet_int();
int main()
{
  int n = nondet_int();
  int j = 0;
  while (j < n)
  {
    j++;
    assert(j < 5);
  }
  int i = 0, s = 0;
  __ESBMC_loop_invariant(i >= 0);
  while (i < 3)
  {
    s++;
    i++;
  }
  assert(s == 3);
  return 0;
}
