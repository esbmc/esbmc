#include <stddef.h>
#define SIZE 5

/* Companion fail case for wildcard_enforce_ptr_pass.
 * The ensures clause claims the return value equals vec[1],
 * but the body returns vec[0], so enforcement must fail. */

int fst(const int vec[SIZE])
{
  __ESBMC_requires(vec != NULL);
  __ESBMC_assigns();
  __ESBMC_ensures(__ESBMC_return_value == vec[1]); /* wrong: should be vec[0] */
  return vec[0];
}

int main()
{
  int vec[SIZE] = {10, 11, 12, 13, 14};
  int res = fst(vec);
  return 0;
}
