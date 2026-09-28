// Everything a macro expands reports the expansion location, so two lambdas in
// one macro body shared a file, line and column; the second reused the first's
// closure record and got an operator() with no body (#7530).
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
  assert(f(5) == (5 + 1) + (5 - 1));
  return 0;
}
