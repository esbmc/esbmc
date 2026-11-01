// Members in another order are another layout: not the same function.
#include <cstdlib>
struct P { int a; int b; };
__ESBMC_logic int sum(const P *p, int n) { return n <= 0 ? 0 : sum(p, n - 1) + p[n - 1].a; }
namespace old_ns {
struct P { int b; int a; };
__ESBMC_logic int sum(const P *p, int n) { return n <= 0 ? 0 : sum(p, n - 1) + p[n - 1].a; }
}
int nondet_int();
int main()
{
  int n = nondet_int();
  __ESBMC_assume(n >= 0 && n < 1000);
  P *p = (P *)malloc(n * sizeof(P));
  __ESBMC_assert(sum(p, n) == old_ns::sum((const old_ns::P *)p, n), "same definition over an isomorphic struct");
  return 0;
}
