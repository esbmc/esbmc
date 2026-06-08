#include <stddef.h>
#define SIZE 5

/* Regression test for GitHub issue #4046:
 * --enforce-contract '*' --function fst was producing a spurious
 * "dereference failure: Incorrect alignment" because the wildcard
 * expansion inserts full IDs ("c:@F@fst") into to_enforce, but
 * --function gives only the short name ("fst"), causing the
 * function_name == entry_function comparison to always be false
 * and skipping pointer-param backing-storage allocation. */

int fst(const int vec[SIZE])
{
  __ESBMC_requires(vec != NULL);
  __ESBMC_requires(0 < SIZE);
  __ESBMC_assigns();
  __ESBMC_ensures(__ESBMC_return_value == vec[0]);
  return vec[0];
}

int main()
{
  int vec[SIZE] = {10, 11, 12, 13, 14};
  int res = fst(vec);
  return 0;
}
