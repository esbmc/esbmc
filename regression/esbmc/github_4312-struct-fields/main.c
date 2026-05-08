#include <assert.h>

// Regression for issue #4312-A: struct nondets must render every
// declared field, not silently drop ones the SMT model left
// unconstrained. Same input as github_4309-struct, but the test
// asserts both .x= and .y= appear in every witness.

struct point
{
  int x;
  int y;
};

int main(void)
{
  struct point p;
  if (p.x < -1 || p.x > 1)
    return 0;
  if (p.y < -1 || p.y > 1)
    return 0;
  int absx = p.x < 0 ? -p.x : p.x;
  int absy = p.y < 0 ? -p.y : p.y;
  assert(absx + absy > 1);
  return 0;
}
