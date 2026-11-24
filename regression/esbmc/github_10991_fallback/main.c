#include <assert.h>
#include <string.h>

unsigned nondet_uint(void);

struct s
{
  char a;
  char b;
};

__attribute__((annotate("__ESBMC_inf_size"))) char pool[1];

int main()
{
  unsigned n = nondet_uint();
  __ESBMC_assume(n <= 2);

  char big[65] = {0};
  memset(big + 63, 5, n);
  assert(n < 2 || big[64] == 5);

  struct s x = {1, 1};
  memset(&x, 0, n);
  assert(n < 2 || x.b == 0);

  unsigned k = nondet_uint();
  __ESBMC_assume(k >= 2 && k <= 3);
  char vla[k];
  memset(vla, 6, n);
  assert(n < 2 || vla[1] == 6);

  memset(pool, 7, n);
  assert(n < 2 || pool[1] == 7);
  return 0;
}
