/* Negative counterpart of github_7597-literal: the loop still has to fold for
 * this claim to be reached, and the symbolic element must not read back as
 * the literal beside it. */
#include <assert.h>

struct pair
{
  int n;
  int r;
};

int nondet_int(void);

int main(void)
{
  int v = nondet_int();
  struct pair s = {5, v};
  int c = 0;

  for (int i = 0; i < s.n; i++)
    c++;

  assert(s.r == 5);
  return 0;
}
