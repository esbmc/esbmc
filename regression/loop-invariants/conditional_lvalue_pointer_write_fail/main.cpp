#include <cassert>

struct S
{
  int e[4];
};

bool nondet_bool();

int main()
{
  S a{}, b{};
  S *p = &a, *q = &b;
  bool c = nondet_bool();
  int n = 4;
  __ESBMC_loop_invariant(n >= 0);
  while (n > 0)
  {
    (c ? *p : *q).e[1] = 7;
    --n;
  }
  assert(a.e[1] == 0);
  return 0;
}
