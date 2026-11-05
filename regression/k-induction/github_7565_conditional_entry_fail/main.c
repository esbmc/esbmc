/* The #7565 residual: the loop's only entry is a *conditional* jump landing
 * past the head, which the step cannot havoc. Unhavoced, the step would prove
 * the program from its concrete entry state at k = 2, so k-induction disables
 * it and the base case finds the violation at k = 50. */

#include <assert.h>

extern int nondet_int(void);

int main()
{
  int i = 0, s = 0, x = 0;
  if (nondet_int())
    goto mid;
  return 0;
top:
  s = s + 1;
mid:
  i = i + 1;
  if (i == 50)
    x = 1;
  assert(x == 0);
  if (i < 100)
    goto top;
  return 0;
}
