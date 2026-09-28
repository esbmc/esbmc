// The two lambdas must stay distinct: their results are 6 and 4, not 6 and 6
// (#7530).
#include <cassert>

static int use(int (*p)(int), int (*q)(int), int x)
{
  return p(x) + q(x);
}

#define PAIR                                                                   \
  [](int i) -> int { return i + 1; }, [](int i) -> int { return i - 1; }

static int f(int x)
{
  return use(PAIR, x);
}

int main()
{
  assert(f(5) == (5 + 1) + (5 + 1));
  return 0;
}
