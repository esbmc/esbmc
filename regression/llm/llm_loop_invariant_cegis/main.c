/* The branch in the body puts this loop out of reach of the affine
 * synthesiser. The scripted model first answers with one wrong and one weak
 * invariant; the feedback leads it to add the counter bound. */
#include <assert.h>

unsigned nondet_uint(void);

int main(void)
{
  unsigned n = nondet_uint();
  unsigned i = 0, x = 0, y = 0;

  while (i < n)
  {
    if (i % 2 == 0)
      x++;
    else
      y++;
    i++;
  }

  assert(x + y == n);
  return 0;
}
