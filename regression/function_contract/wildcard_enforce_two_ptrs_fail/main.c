#include <stddef.h>

/* Companion fail case: ensures claims *a + *b + 1 but body returns *a + *b. */

int add_ptrs(const int *a, const int *b)
{
  __ESBMC_requires(a != NULL);
  __ESBMC_requires(b != NULL);
  __ESBMC_assigns();
  __ESBMC_ensures(__ESBMC_return_value == *a + *b + 1); /* wrong */
  return *a + *b;
}

int main()
{
  int x = 3, y = 4;
  int res = add_ptrs(&x, &y);
  return 0;
}
