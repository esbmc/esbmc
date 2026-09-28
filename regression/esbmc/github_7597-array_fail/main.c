/* Negative counterpart of github_7597-array: the element written the symbolic
 * value must keep it, not the counter's. */
#include <assert.h>

int A[3];

int nondet_int(void);

int main(void)
{
  int in = nondet_int();

  for (A[0] = 1; A[0] <= 5; A[0] = A[0] + 1)
    A[1] = in;

  assert(A[1] == 6);
  return 0;
}
